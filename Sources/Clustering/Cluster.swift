import LASwift
import Foundation
import Accelerate
import CClustering


enum DataPoint {
    case page
    case note
}

enum WhereToAdd {
    case first
    case last
    case middle
}

/// Error enums
enum MatrixError: Error {
    case dimensionsNotMatching
    case matrixNotSquare
    case pageOutOfDimensions
}

public enum AdditionError: Error {
    case moreThanOneObjectToAdd
    case noObjectsToAdd
}

enum CClusteringError: Error {
    case tokenizerInitialization
}

enum AvailableEmbedding {
    case onlyTitle
    case onlyContent
    case titleAndContent
}


class ModelInference {
    lazy var model: UnsafeMutableRawPointer! = {
        /*guard var modelPath = Bundle.module.path(forResource: "minilm_multilingual", ofType: "dylib"),
           var tokenizerModelPath = Bundle.module.path(forResource: "sentencepiece", ofType: "bpe.model")
        else {
          fatalError("Resources not found")
        }*/
        var modelPath = "/Users/jplu/dev/clustering/Sources/Clustering/Resources/minilm_multilingual.dylib"
        var tokenizerModelPath = "/Users/jplu/dev/clustering/Sources/Clustering/Resources/sentencepiece.bpe.model"
        var model: UnsafeMutableRawPointer!
        
        modelPath.withUTF8 { cModelPath in
            tokenizerModelPath.withUTF8 { cTokenizerModelPath in
                model = createModelInferenceWrapper(cModelPath.baseAddress, cTokenizerModelPath.baseAddress)
            }
        }

        return model
    }()
    
    @MainActor func encode(text: String) async throws -> [Double] {
        var content = text
        var result = ModelInferenceResult()
        var ret: Int32 = -1
        
        content.withUTF8 { cText in
            ret = doModelInference(self.model, cText.baseAddress, &result)
        }
            
        if ret == 0 {
            let vector = Array(UnsafeBufferPointer(start: result.weigths, count: Int(result.size)))
            
            return vector.map{Double($0)}
        }
        
        throw CClusteringError.tokenizerInitialization
    }
    
    deinit {
        removeModelInferenceWrapper(self.model)
    }
}


public class SimilarityMatrix {
    var matrix = Matrix([[0]])

    /// Add a data point (webpage or note) to the similarity matrix
    ///
    /// - Parameters:
    ///   - similarities: A vector of similarities between the data point to be added and all existing data points, by the order of appearence in the matrix
    ///   - type: The type of data point, either page or note
    ///   - numExistingNotes: The number of notes already existing in the matrix
    ///   - numExistingPages: The number of pages already existing in the matrix
    func addDataPoint(similarities: [Double], type: DataPoint, numExistingNotes: Int, numExistingPages: Int) throws {
        guard matrix.rows == matrix.cols else {
            throw MatrixError.matrixNotSquare
        }
        
        guard matrix.rows == similarities.count else {
            throw MatrixError.dimensionsNotMatching
        }

        var whereToAdd: WhereToAdd?
        
        if numExistingPages == 0 && numExistingNotes == 0 {
            self.matrix = Matrix([[0]])
        
            return
        } else if type == .page || numExistingPages == 0 {
            whereToAdd = .last
        } else if numExistingNotes == 0 {
            whereToAdd = .first
        } else { // Adding a note, when there's already at least one note and one page
            whereToAdd = .middle
        }

        if let whereToAdd = whereToAdd {
            switch whereToAdd {
            case .first:
                self.matrix = similarities ||| self.matrix
                
                var similarities_row = similarities
                
                similarities_row.insert(0.0, at: 0)
                
                self.matrix = similarities_row === self.matrix
            case .last:
                self.matrix = self.matrix ||| similarities
                
                var similarities_row = similarities
                
                similarities_row.append(0.0)
                
                self.matrix = self.matrix === similarities_row
            case .middle:
                let upLeft = self.matrix ?? (.Take(numExistingNotes), .Take(numExistingNotes))
                let upRight = self.matrix ?? (.Take(numExistingNotes), .TakeLast(numExistingPages))
                let downLeft = self.matrix ?? (.TakeLast(numExistingPages), .Take(numExistingNotes))
                let downRight = self.matrix ?? (.TakeLast(numExistingPages), .TakeLast(numExistingPages))
                let left = upLeft === (Array(similarities[0..<numExistingNotes]) === downLeft)
                let right = upRight === (Array(similarities[numExistingNotes..<similarities.count]) === downRight)
                var longSimilarities = similarities
                
                longSimilarities.insert(0, at: numExistingNotes)
                
                self.matrix = left ||| (longSimilarities ||| right)
            }
        }
    }

    /// Remove a data point (webpage or note, currently used only for pages) from the similarity matrix
    ///
    /// - Parameters:
    ///   - Index: The index of the data point to be removed
    func removeDataPoint(index: Int) throws {
        guard matrix.rows == matrix.cols else {
          throw MatrixError.matrixNotSquare
        }
        
        guard index <= matrix.rows else {
          throw MatrixError.pageOutOfDimensions
        }

        var indecesToKeep = [Int](0..<self.matrix.rows)
        
        indecesToKeep = Array(Set(indecesToKeep).subtracting(Set([index]))).sorted()
        self.matrix = self.matrix ?? (.Pos(indecesToKeep), .Pos(indecesToKeep))
    }
}


// swiftlint:disable:next type_body_length
public class Cluster {
    let thresholdComparison = 0.3
    var pages = [Page]()
    var notes = [ClusteringNote]()
    // As the adjacency matrix is never touched on its own, just through the sub matrices, it does not need add or remove methods.
    var adjacencyMatrix = Matrix([[0]])
    var textualSimilarityMatrix = SimilarityMatrix()
    @MainActor let modelInf = ModelInference()
    
    public init() {}

    ///  Extract a submatrix from a matrix corresponding to a list of indeces to include,
    ///  both rows and columns
    ///
    /// - Parameters:
    ///   - matrix: The original matrix
    ///   - withIndeces: The indeces to include in the new matrix
    /// - Returns: The nre matrix, containing only the desired indeces
    func getSubmatrix(of matrix: Matrix, withIndeces: [Int]) throws -> Matrix {
        guard matrix.rows == matrix.cols else {
            throw MatrixError.dimensionsNotMatching
        }
        
        guard let max = withIndeces.max(), max < matrix.rows else {
            throw MatrixError.pageOutOfDimensions
        }
        
        return matrix ?? (.Pos(withIndeces.sorted()), .Pos(withIndeces.sorted()))
    }
    
    /// Perform spectral clustering over a given adjacency matrix
    ///
    /// - Returns: An array of integers, corresponding to a grouping of all data points. The number given to each group is meaningless, only the grouping itself is meaningful
    // swiftlint:disable:next cyclomatic_complexity function_body_length
    func spectralClustering(on matrix: Matrix? = nil, numGroups: Int? = nil, numNotes: Int? = nil, howDeep: Int = 0) throws -> [Int] {
        var matrixToCluster = self.adjacencyMatrix
        
        if let matrix = matrix {
            matrixToCluster = matrix
        }
        
        let numNotes = numNotes ?? self.notes.count
        
        guard matrixToCluster.rows >= 2 else {
            return zeros(1, matrixToCluster.rows).flat.map { Int($0) }
        }
        
        guard howDeep < 4 else {
            return [Int](repeating: 0, count: matrixToCluster.rows)
        }

        let d = reduce(matrixToCluster, sum, .Row)
        let d1: [Double] = d.map { elem in
            if elem < 1e-5 {
                return elem
            } else {
                return 1 / elem
            }
        }
        
        let D = diag(d)
        let D1 = diag(d1)
        
        // This naming makes sense as D1 is 1/D
        let laplacianNn = D - matrixToCluster
        // Random-walk Laplacian
        let laplacian = D1 * laplacianNn
        let eigen = eig(laplacian)
        var eigenVals = reduce(eigen.D, sum)
        let indices = eigenVals.indices
        let combined = zip(eigenVals, indices).sorted { $0.0 < $1.0 }
        
        eigenVals = combined.map { $0.0 }
        
        let permutation = combined.map { $0.1 }
        var eigenVcts = eigen.V ?? (.All, .Pos(permutation))
        var numClusters: Int
        
        if let numGroups = numGroups {
            numClusters = numGroups
        } else {
            let eigenValsDifference = zip(eigenVals, eigenVals.dropFirst()).map { abs(($1 - $0) / max($0, 0.0001)) }
            let maxDifference = eigenValsDifference.max() ?? 0
        
            numClusters = (eigenValsDifference.firstIndex(of: maxDifference) ?? 0) + 1
        }

        guard numClusters > 1 else {
            return zeros(1, matrixToCluster.rows).flat.map { Int($0) }
        }
        
        if eigenVcts.cols > numClusters {
            eigenVcts = eigenVcts ?? (.All, .Take(numClusters))
        }
        
        var points = [Vector]()
        
        for row in 0..<eigenVcts.rows {
            points.append(Vector(eigenVcts[row: row]))
        }
        
        let labels = [Int](0...numClusters - 1)
        var bestPredictedLabels: [Int] = []
        var bestLabelsScore: Double?
        var predictedLabels: [Int] = []
        let kmeans = KMeans(labels: labels)
        
        for _ in 1...15 {
            predictedLabels = []
            var newLabelScore = 0.0
        
            kmeans.trainCenters(points, convergeDistance: 0.00001)
            
            for point in points {
                let (label, distance) = kmeans.fit(point)
                predictedLabels.append(label)
                newLabelScore += pow(distance, 2.0)
            }
            
            newLabelScore *= Double(1 + numClusters - Set(predictedLabels).count)
            newLabelScore *= Double(self.notes.count - Set(predictedLabels[0..<numNotes]).count)
            
            if bestLabelsScore == nil || newLabelScore < (bestLabelsScore ?? 1000) {
                bestLabelsScore = newLabelScore
                bestPredictedLabels = predictedLabels
            }
        }
        
        let predictedLabelsForNotes = Set(bestPredictedLabels[0..<numNotes])
        
        if predictedLabelsForNotes.count < numNotes {
            for label in predictedLabelsForNotes {
                let indeces = bestPredictedLabels.indices.filter {bestPredictedLabels[$0] == label}
                let newNumNotes = indeces.firstIndex(where: { $0 >= numNotes }) ?? indeces.count
        
                if newNumNotes > 1 && newNumNotes < indeces.count {
                    var newGroups = try self.spectralClustering(on: getSubmatrix(of: matrixToCluster, withIndeces: indeces), numGroups: 2 , numNotes: newNumNotes, howDeep: howDeep + 1)
                
                    if let max = bestPredictedLabels.max() {
                        newGroups = newGroups.map { $0 + max + 1 }
                    }
                    
                    bestPredictedLabels = bestPredictedLabels.enumerated().map { label in
                        if indeces.contains(label.offset) {
                            return newGroups[indeces.firstIndex(of: label.offset)!]
                        } else {
                            return label.element
                        }
                    }
                }
            }
        }
        
        return bestPredictedLabels
    }

    /// Compute the cosine similarity between two vectors
    ///
    /// - Parameters:
    ///   - vector1: a vector
    ///   - vector2: a vector
    /// - Returns: The cosine similarity between the two given vectors.
    func cosineSimilarity(vector1: [Double], vector2: [Double]) -> Double {
        let vec1Normed = cblas_dnrm2(Int32(vector1.count), vector1, 1)
        let vec2Normed = cblas_dnrm2(Int32(vector2.count), vector2, 1)
        let dotProduct = cblas_ddot(Int32(vector1.count), vector1, 1, vector2, 1)
        let res = dotProduct / (vec1Normed * vec2Normed)

        return res
    }
    
    func meanVectors(vec1: [Double], vec2: [Double]) -> [Double] {
        var res: [Double] = []
        
        for (val1, val2) in zip(vec1, vec2) {
            res.append((val1 + val2) / 2)
        }
        
        return res
    }

    /// Compute the cosine similarity of the textual embedding of the current data point
    /// against all existing data point
    ///
    /// - Parameters:
    ///   - textualEmbedding: The embedding of the current data point
    ///   - language: The language of the current data point
    ///   - index: The index of the data point within the corresponding vector (pages or notes)
    ///   - dataPointType: page or note
    ///   - changeContent: Is this a part of a content changing operation (rather than addition)
    /// - Returns: A list of cosine similarity scores
    func scoreSimilarity(embedding: [Double], index: Int, dataPointType: DataPoint, typeEmbedding: AvailableEmbedding) -> [Double] {
        var scores = [Double]()
        for note in notes.enumerated() {
            if dataPointType == .note {
                if note.offset == index {
                    break
                }
                
                scores.append(-1.0)
            } else {
                switch typeEmbedding {
                case .titleAndContent:
                    if !note.element.meanTitleContentEmbedding.isEmpty {
                        scores.append(self.cosineSimilarity(vector1: note.element.meanTitleContentEmbedding, vector2: embedding))
                    } else if !note.element.titleEmbedding.isEmpty {
                        scores.append(self.cosineSimilarity(vector1: note.element.titleEmbedding, vector2: embedding))
                    } else if !note.element.contentEmbedding.isEmpty {
                        scores.append(self.cosineSimilarity(vector1: note.element.contentEmbedding, vector2: embedding))
                    } else {
                        scores.append(0.0)
                    }
                case .onlyContent:
                    if !note.element.contentEmbedding.isEmpty {
                        scores.append(self.cosineSimilarity(vector1: note.element.contentEmbedding, vector2: embedding))
                    } else {
                        scores.append(0.0)
                    }
                case .onlyTitle:
                    if !note.element.titleEmbedding.isEmpty && typeEmbedding == .onlyTitle {
                        scores.append(self.cosineSimilarity(vector1: note.element.titleEmbedding, vector2: embedding))
                    } else {
                        scores.append(0.0)
                    }
                }
            }
        }
        
        for page in pages.enumerated() {
            // The textual vector might be empty, when the OS is not up to date
            // then the score will be 0.0
            if page.offset == index && dataPointType == .page {
                break
            }
            
            switch typeEmbedding {
            case .titleAndContent:
                if !page.element.meanTitleContentEmbedding.isEmpty {
                    scores.append(self.cosineSimilarity(vector1: page.element.meanTitleContentEmbedding, vector2: embedding))
                } else if !page.element.titleEmbedding.isEmpty {
                    scores.append(self.cosineSimilarity(vector1: page.element.titleEmbedding, vector2: embedding))
                } else if !page.element.contentEmbedding.isEmpty {
                    scores.append(self.cosineSimilarity(vector1: page.element.contentEmbedding, vector2: embedding))
                } else {
                    scores.append(0.0)
                }
            case .onlyContent:
                if !page.element.contentEmbedding.isEmpty {
                    scores.append(self.cosineSimilarity(vector1: page.element.contentEmbedding, vector2: embedding))
                } else {
                    scores.append(0.0)
                }
            case .onlyTitle:
                if !page.element.titleEmbedding.isEmpty && typeEmbedding == .onlyTitle {
                    scores.append(self.cosineSimilarity(vector1: page.element.titleEmbedding, vector2: embedding))
                } else {
                    scores.append(0.0)
                }
            }
        }
        
        return scores
    }

    /// Perform the textual similarity process for a new or existing data point
    ///
    /// - Parameters:
    ///   - index: The index of the data point within the corresponding vector (pages or notes)
    ///   - dataPointType: page or note
    ///   - changeContent: Is this a part of a content changing operation (rather than addition)
    func textualSimilarityProcess(index: Int, dataPointType: DataPoint) async throws {
        var content: String
        var title: String
        
        if dataPointType == .page {
            content = pages[index].content
            title = pages[index].title
        } else {
            content = notes[index].content
            title = notes[index].title
        }
        
        var meanTitleContentEmbedding: [Double] = []
        var contentEmbedding: [Double] = []
        var titleEmbedding: [Double] = []
        var finalEmbedding: [Double] = []
        var typeEmbedding: AvailableEmbedding
        var scores: [Double] = []
        
        if !content.isEmpty && !title.isEmpty {
            contentEmbedding = try await self.modelInf.encode(text: content)
            titleEmbedding = try await self.modelInf.encode(text: title)
            
            meanTitleContentEmbedding = self.meanVectors(vec1: contentEmbedding, vec2: titleEmbedding)
            finalEmbedding = meanTitleContentEmbedding
            typeEmbedding = AvailableEmbedding.titleAndContent
        } else if !content.isEmpty && title.isEmpty {
            contentEmbedding = try await self.modelInf.encode(text: content)
            finalEmbedding = contentEmbedding
            typeEmbedding = AvailableEmbedding.onlyContent
        } else {
            titleEmbedding = try await self.modelInf.encode(text: title)
            finalEmbedding = titleEmbedding
            typeEmbedding = AvailableEmbedding.onlyTitle
        }
        
        scores = self.scoreSimilarity(embedding: finalEmbedding, index: index, dataPointType: dataPointType, typeEmbedding: typeEmbedding)
        
        switch dataPointType {
        case .page:
            self.pages[index].meanTitleContentEmbedding = meanTitleContentEmbedding
            self.pages[index].contentEmbedding = contentEmbedding
            self.pages[index].titleEmbedding = titleEmbedding
        case .note:
            self.notes[index].meanTitleContentEmbedding = meanTitleContentEmbedding
            self.notes[index].contentEmbedding = contentEmbedding
            self.notes[index].titleEmbedding = titleEmbedding
        }
        
        if self.pages.count + self.notes.count > 1 {
            try self.textualSimilarityMatrix.addDataPoint(similarities: scores, type: dataPointType, numExistingNotes: max(self.notes.count - 1, 0), numExistingPages: self.pages.count)
        }
    }

    /// Find the location of a specific page in the pages array
    ///
    /// - Parameters:
    ///   - pageID: The ID of the desired page
    /// - Returns: The index of the page in the pages array, if it exists
    func findPageInPages(pageID: UUID) -> Int? {
        let pageIDs = self.pages.map({ $0.id })
        
        return pageIDs.firstIndex(of: pageID)
    }

    /// Find the location of a specific note in the notes array
    ///
    /// - Parameters:
    ///   - noteID: The ID of the desired note
    /// - Returns: The index of the note in the notes array
    func findNoteInNotes(noteID: UUID) -> Int? {
        let noteIDs = self.notes.map({ $0.id })
        
        return noteIDs.firstIndex(of: noteID)
    }

    /// A method that returns all data related to a page or a note for the purpose of exportation
    ///
    /// - Parameters:
    ///   - id: The ID of a page or a note
    /// - Returns: InformationForId struct with all relevant information
    public func getExportInformationForId(id: UUID) -> InformationForId {
        if let pageIndex = findPageInPages(pageID: id) {
            return InformationForId(title: pages[pageIndex].title, content: pages[pageIndex].content)
        } else if let noteIndex = findNoteInNotes(noteID: id) {
            return InformationForId(title: notes[noteIndex].title, content: notes[noteIndex].content)
        } else {
            return InformationForId()
        }
    }

    /// Binarize a matrix according to a threshold (over = 1, under = 0).
    /// This function is used in some of the adjacency matrix construction candidates
    ///
    /// - Parameters:
    ///   - matrix: The matrix to be binarized
    ///   - threshold: The binarization threshold
    /// - Returns: The binarized version of the input matrix
    func binarise(matrix: Matrix) -> Matrix {
        let result = zeros(matrix.rows, matrix.cols)
        
        for row in 0..<matrix.rows {
            for column in 0..<matrix.cols {
                if matrix[row, column] > self.thresholdComparison {
                    result[row, column] = 1.0
                }
            }
        }
        
        return result
    }

    /// Threshold a matrix according to a threshold (over =  original value, under = 0).
    /// This function is used in some of the adjacency matrix construction candidates
    ///
    /// - Parameters:
    ///   - matrix: The matrix to be thresholded
    ///   - threshold: The binarization threshold
    /// - Returns: The thresholded version of the input matrix
    func threshold(matrix: Matrix, threshold: Double) -> Matrix {
        let result = matrix
        
        for row in 0..<matrix.rows {
            for column in 0..<matrix.cols {
                if matrix[row, column] < threshold {
                    result[row, column] = 0.0
                }
            }
        }
        
        return result
    }

    public func removeNote(noteId: UUID) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]], similarities: [UUID: [UUID: Double]]) {
        if let noteIndex = self.findNoteInNotes(noteID: noteId) {
            if self.adjacencyMatrix.rows > 1 {
                try self.textualSimilarityMatrix.removeDataPoint(index: noteIndex)
                
                self.adjacencyMatrix = self.binarise(matrix: self.textualSimilarityMatrix.matrix)
            }
            
            self.notes.remove(at: noteIndex)
            
            if self.notes.count > 0 {
                let predictedClusters = try self.spectralClustering()
                let stablizedClusters = self.stabilize(predictedClusters)
                let (resultPages, resultNotes) = self.clusterizeIDs(labels: stablizedClusters)
                let similarities = self.createSimilarities(pageGroups: resultPages, noteGroups: resultNotes)

                return (pageGroups: resultPages, noteGroups: resultNotes, similarities: similarities)
            }
        }
        
        return (pageGroups: [], noteGroups: [], similarities: [:])
    }
    
    public func removePage(pageId: UUID) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]], similarities: [UUID: [UUID: Double]]) {
        if let pageIndex = self.findPageInPages(pageID: pageId) {
            if self.adjacencyMatrix.rows > 1 {
                try self.textualSimilarityMatrix.removeDataPoint(index: pageIndex)
                
                self.adjacencyMatrix = self.binarise(matrix: self.textualSimilarityMatrix.matrix)
            }
            
            self.pages.remove(at: pageIndex)
            
            if self.pages.count > 0 {
                let predictedClusters = try self.spectralClustering()
                let stablizedClusters = self.stabilize(predictedClusters)
                let (resultPages, resultNotes) = self.clusterizeIDs(labels: stablizedClusters)
                let similarities = self.createSimilarities(pageGroups: resultPages, noteGroups: resultNotes)

                return (pageGroups: resultPages, noteGroups: resultNotes, similarities: similarities)
            }
        }
        
        return (pageGroups: [], noteGroups: [], similarities: [:])
    }
    
    /// A function to calculate the similarities of notes and active sources with all sources that are in the same group,
    /// for the purpose of using this in score calculation
    ///
    /// - Parameters:
    ///   - pageGroups: array of array of all pages clustered into groups
    ///   - noteGroups: array of array of all notes clustered into corresponding groups
    ///   - activeSources: array of all active sources
    /// - Returns: A dictionary of all notes and active sources (keys), the value for each is a dictionary of all pages
    ///             in the same group (keys) and the corresponding similarity (value)
    func createSimilarities(pageGroups: [[UUID]], noteGroups: [[UUID]]) -> [UUID: [UUID: Double]] {
        var similarities = [UUID: [UUID: Double]]()
        
        // Start by including similarities with notes
        for (pageGroup, noteGroup) in zip(pageGroups, noteGroups) {
            for noteId in noteGroup {
                if let noteIndex = self.findNoteInNotes(noteID: noteId) {
                    similarities[noteId] = [UUID: Double]()
                    
                    for pageId in pageGroup {
                        if let pageIndex = self.findPageInPages(pageID: pageId) {
                            similarities[noteId]?[pageId] = self.textualSimilarityMatrix.matrix[noteIndex, pageIndex + self.notes.count]
                        }
                    }
                }
            }
            
            for pageId in pageGroup {
                if let pageIndex = self.findPageInPages(pageID: pageId) {
                   similarities[pageId] = [UUID:Double]()
                   
                   for suggestedPageId in pageGroup.filter({ $0 != pageId }) {
                       if let suggestedPageIndex = self.findPageInPages(pageID: suggestedPageId) {
                           similarities[pageId]?[suggestedPageId] = self.textualSimilarityMatrix.matrix[pageIndex + self.notes.count, suggestedPageIndex + self.notes.count]
                       }
                   }
                }
            }
        }
        
        return similarities
    }

    /// A function to add a data point to all sub matrices (navigation, textual and entity-based)
    ///
    /// - Parameters:
    ///   - page: The page to be added, in case the data point is a page
    ///   - note: The note to be added, in case the data point is a note
    ///   - replaceContent: A flag to declare that the request comes from PnS, the page
    ///                     already exists, and the text considered for the page should be replaced.
    func updateSubMatrices(page: Page? = nil, note: ClusteringNote? = nil) async throws {
        // Decide if the received data point in a page or a note
        var dataPointType: DataPoint = .note
        
        if page != nil {
            dataPointType = .page
        }
        
        var newIndex = self.pages.count
        
        if let page = page {
            self.pages.append(page)
        }
        
        if let note = note {
            newIndex = self.notes.count
            
            self.notes.append(note)
        }
        // Add to submatrices
        try await self.textualSimilarityProcess(index: newIndex, dataPointType: dataPointType)
    }
    
    /// The main function to access the package, adding a data point (page or note)
    /// to the clustering process. The function makes sure that a request to add a data point
    /// includes an unknown datapoint, in the contrary case it updates the existing
    /// data point
    ///
    /// - Parameters:
    ///   - page: The page to be added, in case the data point is a page
    ///   - note: The note to be added, in case the data point is a note
    ///   - ranking: A ranking of all pages, for the purposes of removing pages when necessary
    ///   - replaceContent: A flag to declare that the request comes from PnS, the page
    ///                     already exists, and the text considered for the page should be replaced.
    /// - Returns: (through completion)
    ///             - pageGroups: Array of arrays of all pages clustered into groups
    ///             - noteGroups: Array of arrays of all notes clustered into groups, corresponding to the groups of pages
    ///             - sendRanking: A flag to ask the clusteringManager to send page ranking with the next 'add' request, for the purpose of removing some pages
    // swiftlint:disable:next cyclomatic_complexity function_body_length large_tuple
    public func add(page: Page? = nil, note: ClusteringNote? = nil) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]], similarities: [UUID: [UUID: Double]]) {
        // Check that we are adding exactly one object
        if page != nil && note != nil {
            throw AdditionError.moreThanOneObjectToAdd
        }
            
        if page == nil && note == nil {
            throw AdditionError.noObjectsToAdd
        }
            
        // Updating all sub-matrices
        try await self.updateSubMatrices(page: page, note: note)
                
        self.adjacencyMatrix = self.binarise(matrix: self.textualSimilarityMatrix.matrix)
        
        let predictedClusters = try self.spectralClustering()
        let stablizedClusters = self.stabilize(predictedClusters)
        let (resultPages, resultNotes) = self.clusterizeIDs(labels: stablizedClusters)
        let similarities = self.createSimilarities(pageGroups: resultPages, noteGroups: resultNotes)

        return (pageGroups: resultPages, noteGroups: resultNotes, similarities: similarities)
    }

    /// Stabilize clustering results to maintain order from one clustering to another
    ///
    /// - Parameters:
    ///   - predictedClusteris: The result of spectral clustering
    /// - Returns: The same clustering result, with order in the labels (0 then 1 etc, by order)
    func stabilize(_ predictedClusters: [Int]) -> [Int] {
        var nextNewCluster = 0
        var clustersMap = [Int: Int]()
        var newClusters = [Int]()
        
        for oldLabel in predictedClusters {
            if let newLabel = clustersMap[oldLabel] {
                newClusters.append(newLabel)
            } else {
                clustersMap[oldLabel] = nextNewCluster
                
                newClusters.append(nextNewCluster)
                
                nextNewCluster += 1
            }
        }
        
        return newClusters
    }

    /// Turn raw clustering result to arrays of arrays for pages and notes
    ///
    /// - Parameters:
    ///   - labels: A stabilized list of labels, corresponding to all data points
    /// - Returns: - An array of arrays of all pages, by groups
    ///            - An array of arrays of all notes, by groups corresponding to the pages
    private func clusterizeIDs(labels: [Int]) -> ([[UUID]], [[UUID]]) {
        guard self.notes.count + self.pages.count > 0 else {
            return ([[UUID]](), [[UUID]]())
        }
        
        var clusterizedPages = [[UUID]]()
        
        if labels.count > 0 {
            for _ in 0...(labels.max() ?? 0) {
                clusterizedPages.append([UUID]())
            }
        }
        
        var clusterizedNotes = [[UUID]]()
        
        if labels.count > 0 {
            for _ in 0...(labels.max() ?? 0) {
                clusterizedNotes.append([UUID]())
            }
        }
        
        for label in labels.enumerated() {
            if label.offset < self.notes.count {
                clusterizedNotes[label.element].append(self.notes[label.offset].id)
            } else {
                clusterizedPages[label.element].append(self.pages[label.offset - self.notes.count].id)
            }
        }
        
        return (clusterizedPages, clusterizedNotes)
    }

    
    // swiftlint:disable:next file_length
}
