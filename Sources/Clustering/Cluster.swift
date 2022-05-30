import LASwift
import Foundation
import NaturalLanguage
import Accelerate
import CClustering

// swiftlint:disable:next type_body_length
public class Cluster {

    enum FindEntitiesIn {
        case content
        case title
    }

    enum AllWeights {
        case navigation
        case text
        case entities
    }

    enum DataPoint {
        case page
        case note
    }

    enum WhereToAdd {
        case first
        case last
        case middle
    }

    public enum Flag {
        case sendRanking
        case addNotes
        case none
    }

    /// enums to control candidates
    enum LaplacianCandidate {
        case nonNormalizedLaplacian
        case randomWalkLaplacian
        case symetricLaplacian
    }

    enum SimilarityMatrixCandidate {
        case navigationMatrix
        case combinationAllSimilarityMatrix
        case combinationAllBinarisedMatrix
        case combinationSigmoidWithTextErasure
        case fixedPagesTestNotes
    }

    enum SimilarityForNotesCandidate {
        case nothing
        case fixed
        case combinationBeforeSigmoid
        case combinationAfterSigmoid
        case sigmoidOnEntities
    }

    enum NumClusterComputationCandidate {
        case threshold
        case biggestDistanceInPercentages
        case biggestDistanceInAbsolute
    }

    /// Error enums
    enum MatrixError: Error {
        case dimensionsNotMatching
        case matrixNotSquare
        case pageOutOfDimensions
    }

    enum CandidateError: Error {
        case unknownCandidate
    }

    enum MatrixTypeError: Error {
        case unknownMatrixType
    }

    public enum AdditionError: Error {
        case moreThanOneObjectToAdd
        case noObjectsToAdd
        case notEnoughTextInNote
        case skippingToNextAddition
        case abortingAdditionDuringClustering
    }

    let additionQueue = DispatchQueue(label: "additionQueue")
    let clusteringQueue = DispatchQueue(label: "clusteringQueue")
    let mainQueue: DispatchQueue
    var additionsInQueue = 0
    var pages = [Page]()
    var notes = [ClusteringNote]()
    // As the adjacency matrix is never touched on its own, just through the sub matrices, it does not need add or remove methods.
    var adjacencyMatrix = Matrix([[0]])
    var navigationMatrix = NavigationMatrix()
    var textualSimilarityMatrix = SimilarityMatrix()
    var entitiesMatrix = SimilarityMatrix()
    var beTogetherMatrix = SimilarityMatrix()
    var beApartMatrix = SimilarityMatrix()
    let tagger = NLTagger(tagSchemes: [.nameType])
    let entityOptions: NLTagger.Options = [.omitPunctuation, .omitWhitespace, .joinNames]
    let entityTags: [NLTag] = [.personalName, .placeName, .organizationName]
    var timeToRemove: Double = 3 // If clustering takes more than this (in seconds) we start removing pages
    let titleSuffixes = [" - Google Search", " - YouTube", "| Ebay", "| eBay", " - Ecosia", " at DuckDuckGo", " - Bing", " | The Guardian"]
    let titlePrefixes = ["Amazon.com"]
    let beta = 50.0
    var noteContentThreshold: Int
    //let extractor = JusText()
    var pagesWithContent = 0
    var askForNotes = false
    var skippedPages = [Page]()
    var skippedNotes = [ClusteringNote]()
    
    var isClustering: Bool = false {
        didSet {
             if !isClustering {
                 for page in skippedPages {
                     self.add(page: page, ranking: nil) { _ in }
                 }
                 self.skippedPages = []
                 for note in skippedNotes {
                     self.add(note: note, ranking: nil) { _ in }
                 }
                 self.skippedNotes = []
             }
        }
    }


    //Define which Laplacian to use
    var laplacianCandidate = LaplacianCandidate.randomWalkLaplacian
    // Define which similarity matrix to use
    var matrixCandidate = SimilarityMatrixCandidate.combinationSigmoidWithTextErasure
    var noteMatrixCandidate = SimilarityForNotesCandidate.fixed
    // Define which number of clusters computation to use
    var numClustersCandidate = NumClusterComputationCandidate.biggestDistanceInPercentages
    var candidate: Int
    var weights = [AllWeights: Double]()

    public init(candidate: Int = 2, weightNavigation: Double = 0.5, weightText: Double = 0.9, weightEntities: Double = 0.2, noteContentThreshold: Int = 100) {
        mainQueue = DispatchQueue(label: "ClusteringModel")
        self.candidate = candidate
        self.weights[.navigation] = weightNavigation
        self.weights[.text] = weightText
        self.weights[.entities] = weightEntities
        self.noteContentThreshold = noteContentThreshold
    
        do {
            try self.performCandidateChange()
        } catch {
            fatalError()
        }
    }
           
    deinit {
        removeModelInferenceWrapper(modelinf)
    }

    lazy var modelinf: UnsafeMutableRawPointer! = {
        /*guard var modelPath = Bundle.module.path(forResource: "minilm_multilingual", ofType: "dylib"),
           var tokenizerModelPath = Bundle.module.path(forResource: "sentencepiece", ofType: "bpe.model")
        else {
          fatalError("Resources not found")
        }*/
        var modelPath = "/Users/jplu/dev/clustering/Sources/Clustering/Resources/minilm_multilingual.dylib"
        var tokenizerModelPath = "/Users/jplu/dev/clustering/Sources/Clustering/Resources/sentencepiece.bpe.model"

        var modelinf: UnsafeMutableRawPointer!
        
        mainQueue.sync {
          modelPath.withUTF8 { cModelPath in
            tokenizerModelPath.withUTF8 { cTokenizerModelPath in
              modelinf = createModelInferenceWrapper(cModelPath.baseAddress, cTokenizerModelPath.baseAddress)
            }
          }
        }

        return modelinf
    }()

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

    public class NavigationMatrix: SimilarityMatrix {

        /// Remove a data point (webpage or note, currently used only for pages) from the navigation similarity matrix - create navigation links between incoming and outgoing links begore removing
        ///
        /// - Parameters:
        ///   - Index: The index of the data point to be removed
         override func removeDataPoint(index: Int) throws {
             var connectionsVct = (self.matrix ?? (.Pos([index]), .All)).flat
             connectionsVct.remove(at: index)
             try super.removeDataPoint(index: index)
             for i in 0..<self.matrix.rows where connectionsVct[i] == 1.0 {
                 for j in 0..<self.matrix.rows where i != j && connectionsVct[j] == 1.0 {
                     self.matrix[i, j] = 1.0
                     self.matrix[j, i] = 1.0
                 }
             }
         }
     }

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
            if elem < 1e-5 { return elem } else { return 1 / elem }
        }
        let D = diag(d)
        let D1 = diag(d1)
        // This naming makes sense as D1 is 1/D
        let laplacianNn = D - matrixToCluster
        // This is the 'simplest' non-normalized Laplacian

        var laplacian: Matrix
        switch self.laplacianCandidate { // This switch takes care of the choice of Laplacian
        case .nonNormalizedLaplacian: // Non-normalised graph Laplacian
            laplacian = laplacianNn
        case .randomWalkLaplacian: // Random-walk Laplacian
            laplacian = D1 * laplacianNn
        case .symetricLaplacian: // Symmetric Laplacian
            laplacian = sqrt(D1) * laplacianNn * sqrt(D1)
        }
        //TODO: If necessary, add the Laplacian of Zelnik-Manor

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
            switch self.numClustersCandidate { // This switch takes care of the number of total classes
            case .threshold: // Threshold
                eigenVals.removeAll(where: { $0 > 1e-5 })
                numClusters = eigenVals.count
            case .biggestDistanceInPercentages: // Biggest distance in percentages
                let eigenValsDifference = zip(eigenVals, eigenVals.dropFirst()).map { abs(($1 - $0) / max($0, 0.0001)) }
                let maxDifference = eigenValsDifference.max() ?? 0
                numClusters = (eigenValsDifference.firstIndex(of: maxDifference) ?? 0) + 1
            case .biggestDistanceInAbsolute: // Biggest distance, absolute
                var eigenValsDifference = zip(eigenVals, eigenVals.dropFirst()).map { abs(($1 - $0)) }
                eigenValsDifference = eigenValsDifference.map { ($0 * 100).rounded() / 100 }
                let maxDifference = eigenValsDifference.max() ?? 0
                numClusters = (eigenValsDifference.firstIndex(of: maxDifference) ?? 0) + 1
            }
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

    ///  Compute the embedding of the given piece of text if the language of the text is detectable
    ///  and if the OS is at least MacOS 11 and iOS 14.
    ///
    /// - Parameters:
    ///   - text: The text that will be turned into a contextual vector (embedding)
    /// - Returns: The embedding of the given piece of text as an optional and the dominating language of the text, as an optional.
    func textualEmbeddingComputationWithNLEmbedding(text: String/*, language: NLLanguage*/, completion: @escaping ([Double]?)->()) {
        DispatchQueue.main.async {
            if text.isEmpty {
                completion(nil)
            }
            
            var content = text
            var model_result = ModelInferenceResult()
            var ret: Int32 = -1
                
            content.withUTF8 { cText in
                ret = doModelInference(self.modelinf, cText.baseAddress, &model_result)
            }
                
            if ret == 0 {
                let vector = Array(UnsafeBufferPointer(start: model_result.weigths, count: Int(model_result.size)))
                let dvector = vector.map{Double($0)}

                completion(dvector)
            } else {
                completion(nil)
            }
        }
    }

    ///  Finds entities of the specified types in a given text
    ///
    /// - Parameters:
    ///   - text: The text to find entities in
    /// - Returns: An EntitiesInText struct, containing all the entities found in the text
    func findEntitiesInText(text: String) -> EntitiesInText {
        var entitiesInCurrentText = EntitiesInText()
        self.tagger.string = text
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .nameType, options: entityOptions) { tag, tokenRange in
            // Get the most likely tag, and include it if it's a named entity.
            if let tag = tag, entityTags.contains(tag), let contains = entitiesInCurrentText.entities[tag.rawValue]?.contains(String(text[tokenRange]).lowercased()), contains == false {
                entitiesInCurrentText.entities[tag.rawValue]?.append(String(text[tokenRange]).lowercased())
            }
            return true
        }
        return entitiesInCurrentText
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
    func scoreTextualSimilarity(textualEmbedding: [Double]?, /*language: NLLanguage,*/ index: Int, dataPointType: DataPoint, changeContent: Bool = false) -> [Double] {
        var scores = [Double]()
         for note in notes.enumerated() {
            if dataPointType == . note {
                if !changeContent && note.offset == index { break }
                scores.append(-1.0)
            } else if let textualVectorID = note.element.textEmbedding,
                      //let textLanguage = note.element.language,
                      let textualEmbedding = textualEmbedding {//,
                      //textLanguage == language {
                scores.append(self.cosineSimilarity(vector1: textualVectorID, vector2: textualEmbedding))
            } else {
                scores.append(0.0)
            }
        }

        for page in pages.enumerated() {
            // The textual vector might be empty, when the OS is not up to date
            // then the score will be 0.0
            if page.offset == index && dataPointType == .page {
                if changeContent {
                    scores.append(0.0)
                } else { break }
            /*} else if let textLanguage = page.element.language,
                      textLanguage == language {
                if let textualVectorID = page.element.textEmbedding,
                   let textualEmbedding = textualEmbedding {
                    scores.append(self.cosineSimilarity(vector1: textualVectorID, vector2: textualEmbedding))
                } else {
                    scores.append(0.0)
                }
            }*/
            } else if let textualVectorID = page.element.textEmbedding,
                      let textualEmbedding = textualEmbedding {
                scores.append(self.cosineSimilarity(vector1: textualVectorID, vector2: textualEmbedding))
            /*} else if dataPointType == .page {
                scores.append(0.0)
//                scores.append(1.0) // We don't want to "break" connections between langauges*/
            } else {
                scores.append(0.0)
            }
        }
        return scores
    }

    /// Compute the entity similarity between the current data point to be added/changed and all existing data points. This is a new version that compares entities across both titles and text content without seperating them
    ///
    /// - Parameters:
    ///   - entitiesInText: The entities found in the current text
    ///   - index: The index of the data point within the corresponding vector (pages or notes)
    ///   - dataPointType: page or note
    ///   - changeContent: Is this a part of a content changing operation (rather than addition)
    ///   - domain : The domain of the page, in case it is indeed a page
    /// - Returns: A list of similarity scores
    func scoreEntitySimilaritiesInFullDataPoint(entitiesInNewDataPoint: EntitiesInText, index: Int, dataPointType: DataPoint, changeContent: Bool = false, domain: String? = nil) -> [Double] {
        var scores = [Double]()
        for note in notes.enumerated() {
            if dataPointType == .note {
                if !changeContent && note.offset == index { break }
                scores.append(0.0)
            } else {
                let entitiesInNote = (note.element.entities ?? EntitiesInText()) + (note.element.entitiesInTitle ?? EntitiesInText())
                if !entitiesInNote.isEmpty {
                    scores.append(self.jaccardEntities(entitiesText1: entitiesInNewDataPoint, entitiesText2: entitiesInNote))
                } else {
                scores.append(0.0)
                }
            }
        }
        for page in pages.enumerated() {
            if page.offset == index  && dataPointType == .page {
                if changeContent {
                    scores.append(0.0)
                } else { break }
            } else {
                let entitiesInPage = (page.element.entities ?? EntitiesInText()) + (page.element.entitiesInTitle ?? EntitiesInText())
                if !entitiesInPage.isEmpty {
                    var commonDomainTokens: [String]?
                    if let domain = domain,
                       let otherDomain = page.element.domain,
                       domain == otherDomain {
                        commonDomainTokens = Array(Set(domain.components(separatedBy: CharacterSet(charactersIn: "./")) + otherDomain.components(separatedBy: CharacterSet(charactersIn: "./"))))
                    }
                    scores.append(self.jaccardEntities(entitiesText1: entitiesInNewDataPoint, entitiesText2: entitiesInPage, domainTokens: commonDomainTokens))
                } else { scores.append(0.0) }
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
    func textualSimilarityProcess(index: Int, dataPointType: DataPoint, changeContent: Bool = false) throws {
        var content: String?
        var title: String?
        //var language: NLLanguage?
        var scores = [Double](repeating: 0.0, count: self.textualSimilarityMatrix.matrix.rows)
        if dataPointType == .page {
            content = pages[index].cleanedContent
            //language = pages[index].language
            title = pages[index].title
        } else {
            content = notes[index].cleanedContent
            //language = notes[index].language
            title = notes[index].title
        }
        if let content = content/*,
           let language = language*/ {
            let semaphore = DispatchSemaphore(value: 0)
            let title = title ?? ""
            
            self.textualEmbeddingComputationWithNLEmbedding(text: (title + " " + content).trimmingCharacters(in: .whitespacesAndNewlines)/*, language: language*/) { [weak self] result in
                guard let self = self else {
                    return
                }

                scores = self.scoreTextualSimilarity(textualEmbedding: result, /*language: language,*/ index: index, dataPointType: dataPointType, changeContent: changeContent)

                switch dataPointType {
                case .page:
                    self.pages[index].textEmbedding = result
                case .note:
                    self.notes[index].textEmbedding = result
                }
                semaphore.signal()
            }
            semaphore.wait()
        }
        if changeContent {
            var indexToChange = index
            if dataPointType == .page {
                indexToChange += self.notes.count
            }
            self.textualSimilarityMatrix.matrix[row: indexToChange] = scores
            self.textualSimilarityMatrix.matrix[col: indexToChange] = scores
        } else if self.pages.count + self.notes.count > 1 {
            try self.textualSimilarityMatrix.addDataPoint(similarities: scores, type: dataPointType, numExistingNotes: max(self.notes.count - 1, 0), numExistingPages: self.pages.count)
        }
    }

    /// Perform the entity process for a new or existing data point
    ///
    /// - Parameters:
    ///   - index: The index of the data point within the corresponding vector (pages or notes)
    ///   - dataPointType: page or note
    ///   - changeContent: Is this a part of a content changing operation (rather than addition)
    // swiftlint:disable:next cyclomatic_complexity
    func entitiesProcess(index: Int, dataPointType: DataPoint, changeContent: Bool = false) throws {
        var content: String?
        var title: String?
        var domain: String?
        switch dataPointType {
        case .page:
            content = self.pages[index].cleanedContent
            title = self.pages[index].title
            domain = self.pages[index].domain
        case .note:
            content = self.notes[index].cleanedContent
            title = self.notes[index].title
        }
        var entitiesInNewText = EntitiesInText()
        var entitiesInNewTitle = EntitiesInText()
        if let content = content {
            entitiesInNewText = self.findEntitiesInText(text: content)
            switch dataPointType {
            case .page:
                pages[index].entities = entitiesInNewText
            case .note:
                notes[index].entities = entitiesInNewText
            }
        }
        if let title = title {
            entitiesInNewTitle = findEntitiesInText(text: (title.split(separator: " ").map { $0.capitalized }).joined(separator: " ") + " and some text")
            switch dataPointType {
            case .page:
                pages[index].entitiesInTitle = entitiesInNewTitle
            case .note:
                notes[index].entitiesInTitle = entitiesInNewTitle
            }
        }
        let entitiesInNewDataPoint = entitiesInNewText + entitiesInNewTitle
        let scores = self.scoreEntitySimilaritiesInFullDataPoint(entitiesInNewDataPoint: entitiesInNewDataPoint, index: index, dataPointType: dataPointType, changeContent: changeContent, domain: domain)

        if changeContent {
            var indexToChange = index
            if dataPointType == .page {
                indexToChange += self.notes.count
            }
            self.entitiesMatrix.matrix[row: indexToChange] = scores
            self.entitiesMatrix.matrix[col: indexToChange] = scores
        } else if self.pages.count + self.notes.count > 1 {
            try self.entitiesMatrix.addDataPoint(similarities: scores, type: dataPointType, numExistingNotes: max(self.notes.count - 1, 0), numExistingPages: self.pages.count)
        }
    }

    /// Prepare entities of type "PersonName" to be compared
    ///
    /// - Parameters:
    ///   - namesFound: An array of all the PersonName entities found in a text
    /// - Returns: An array of names, each itself an array of all elements available in the name
    func preparePersonName(namesFound: [String]) -> [[String]] {
        let splitNames = namesFound.map { $0.lowercased().components(separatedBy: " ") }
        var preparedNames = [[String]]()
        for name in splitNames {
            outerSwitch: switch name.count {
            case 0:
                break
            case 1:
                for otherName in splitNames where otherName != name {
                    if otherName.contains(name[0]) {
                        break outerSwitch
                    }
                }
                preparedNames.append(name)
            default:
                preparedNames.append(name)
            }
        }
        return preparedNames
    }

    /// A function to compare to sets of PersonalName arrays only
    ///
    /// - Parameters:
    ///   - names1: An array of all the PersonalName entities found in the first data point
    ///   - names2: An array of all the PersonalName entities found in the second data point
    /// - Returns: A similarity score
    func comparePersonNames(names1: [[String]], names2: [[String]]) -> Double {
        var score = 0.0
        for name1 in names1 {
            if name1.count == 1  {
                if names2.contains(name1) {
                    score = max(pow(Double(name1[0].count) / 6.0, 2.0), 1.0)
                } else {
                    for name2 in names2 {
                        if name2.contains(name1[0]) {
                            score = max(pow(Double(name1[0].count) / 6.0, 2.0), 1.0)
                            break
                        }
                    }
                }
            } else {
                if names2.contains(name1) {
                    score = 1
                } //TODO: Add the case where one word is identical and the others are not
            }
        }
        return score
    }

    /// Given a struct of entities that were find, this function removes those that match with tokens from the domain. It is called only when the domain is the same for two pages
    ///
    /// - Parameters:
    ///   - namesFound: An array of all the PersonName entities found in a text
    /// - Returns: An array of names, each itself an array of all elements available in the name
    func removeDomainFromEntities(entitiesInText: EntitiesInText, domainTokens: [String]) -> EntitiesInText {
        var newEntitiesInText = EntitiesInText()
        let domainTokensProcessed = Set(domainTokens.map { $0.lowercased() })
        for entityType in ["PersonalName", "PlaceName", "OrganizationName"] {
            newEntitiesInText.entities[entityType] = (entitiesInText.entities[entityType] ?? []).filter { !domainTokensProcessed.contains($0) }
        }
        return newEntitiesInText
    }

    /// Score the similarity between two EntitiesInText structs, using a modified Jaccard similarity
    ///
    /// - Parameters:
    ///   - entitiesInText1: The entities found in the first text
    ///   - entitiesInText1: The entities found in the first text
    ///   - domainTokens: All the tokens in the domain, in case both data points have the same domain
    /// - Returns: The similarity between the two data points, between 0 and 1
    func jaccardEntities(entitiesText1: EntitiesInText, entitiesText2: EntitiesInText, domainTokens: [String]? = nil) -> Double {
        var entitiesWithoutDomain1 = entitiesText1
        var entitiesWithoutDomain2 = entitiesText2
        if let domainTokens = domainTokens {
            entitiesWithoutDomain1 = self.removeDomainFromEntities(entitiesInText: entitiesText1, domainTokens: domainTokens)
            entitiesWithoutDomain2 = self.removeDomainFromEntities(entitiesInText: entitiesText2, domainTokens: domainTokens)
        }
        let preparedNames1 = self.preparePersonName(namesFound: entitiesWithoutDomain1.entities["PersonalName"] ?? [])
        let preparedNames2 = self.preparePersonName(namesFound: entitiesWithoutDomain2.entities["PersonalName"] ?? [])
        let scorePersonNames = self.comparePersonNames(names1: preparedNames1, names2: preparedNames2)
        var numEntities1 = preparedNames1.count
        var numEntities2 = preparedNames2.count

        var totalEntitiesNoPerson1 = [String]()
        var totalEntitiesNoPerson2 = [String]()
        for entityType in ["PlaceName", "OrganizationName"] {
                totalEntitiesNoPerson1 += (entitiesWithoutDomain1.entities[entityType] ?? [String]()).map { $0.trimmingCharacters(in: .punctuationCharacters) }
                totalEntitiesNoPerson2 += (entitiesWithoutDomain2.entities[entityType] ?? [String]()).map { $0.trimmingCharacters(in: .punctuationCharacters) }
        }

        numEntities1 += Set(totalEntitiesNoPerson1).count
        numEntities2 += Set(totalEntitiesNoPerson2).count
        let maximumEntities = max(numEntities1, numEntities2)
        if maximumEntities > 0 {
            return (Double(Set(totalEntitiesNoPerson1).intersection(Set(totalEntitiesNoPerson2)).count) + scorePersonNames) / Double(maximumEntities)
        } else {
            return 0
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
            return InformationForId(title: pages[pageIndex].title, cleanedContent: pages[pageIndex].cleanedContent, entitiesInText: pages[pageIndex].entities, entitiesInTitle: pages[pageIndex].entitiesInTitle, /*language: pages[pageIndex].language,*/ parentId: pages[pageIndex].parentId)
        } else if let noteIndex = findNoteInNotes(noteID: id) {
            return InformationForId(title: notes[noteIndex].title, cleanedContent: notes[noteIndex].cleanedContent, entitiesInText: notes[noteIndex].entities, entitiesInTitle: notes[noteIndex].entitiesInTitle/*, language: notes[noteIndex].language*/)
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
    func binarise(matrix: Matrix, threshold: Double) -> Matrix {
        let result = zeros(matrix.rows, matrix.cols)
        for row in 0..<matrix.rows {
            for column in 0..<matrix.cols {
                if matrix[row, column] > threshold {
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

    /// Perform an element-wise Sigmoid function on the elements of a matrix
    /// This function is used in some of the adjacency matrix construction candidates
    ///
    /// - Parameters:
    ///   - matrix: The matrix to perform the Sigmoid function on
    ///   - middle: The middle point of the Sigmoid function (sometimes referred to as mu)
    ///   - beta: The steepness of the Sigmoid function
    /// - Returns: The new version of the input matrix
    func performSigmoidOn(matrix: Matrix, middle: Double, beta: Double) -> Matrix {
        return 1 ./ (1 + exp(-beta .* (matrix - middle)))
    }

    /// Create the final adjacency matrix, to be used as an input to spectral clustering
    // swiftlint:disable:next cyclomatic_complexity
    func createAdjacencyMatrix() {
        // Prepare the entire matrix, as if it is composed only of pages
        switch self.matrixCandidate {
        case .navigationMatrix:
            self.adjacencyMatrix = self.navigationMatrix.matrix
        case .combinationAllSimilarityMatrix:
            self.adjacencyMatrix = (self.weights[.text] ?? 0.5) .*  self.textualSimilarityMatrix.matrix + (self.weights[.entities] ?? 0.5) .* self.entitiesMatrix.matrix + (self.weights[.navigation] ?? 0.5) .* self.navigationMatrix.matrix
        case .combinationAllBinarisedMatrix:
            self.adjacencyMatrix = self.binarise(matrix: self.navigationMatrix.matrix, threshold: (self.weights[.navigation] ?? 0.5)) + self.binarise(matrix: self.textualSimilarityMatrix.matrix, threshold: (weights[.text] ?? 0.5)) + self.binarise(matrix: self.entitiesMatrix.matrix, threshold: (weights[.entities] ?? 0.5))
        case .combinationSigmoidWithTextErasure:
//            let navigationSigmoidMatrix = self.performSigmoidOn(matrix: self.navigationMatrix.matrix, middle: self.weights[.navigation] ?? 0.5, beta: self.beta)
            let textSigmoidMatrix = self.performSigmoidOn(matrix: self.textualSimilarityMatrix.matrix, middle: self.weights[.text] ?? 0.5, beta: self.beta)
            let entitySigmoidMatrix = self.performSigmoidOn(matrix: self.entitiesMatrix.matrix, middle: self.weights[.entities] ?? 0.5, beta: self.beta)
//            self.adjacencyMatrix = textSigmoidMatrix .* navigationSigmoidMatrix + entitySigmoidMatrix
            self .adjacencyMatrix = (textSigmoidMatrix .* self.navigationMatrix.matrix + entitySigmoidMatrix) .* self.beApartMatrix.matrix + self.beTogetherMatrix.matrix
        case .fixedPagesTestNotes:
            let navigationSigmoidMatrix = self.performSigmoidOn(matrix: self.navigationMatrix.matrix, middle: 0.5, beta: self.beta)
            let textSigmoidMatrix = self.performSigmoidOn(matrix: self.textualSimilarityMatrix.matrix, middle: 0.9, beta: self.beta)
            let entitySigmoidMatrix = self.performSigmoidOn(matrix: self.entitiesMatrix.matrix, middle: 0.2, beta: self.beta)
            let adjacencyForPages = textSigmoidMatrix .* navigationSigmoidMatrix + entitySigmoidMatrix
            self.adjacencyMatrix = adjacencyForPages
        }

        // Make adjustments for notes
        guard self.notes.count > 0 else { return }
        var adjacencyForNotes: Matrix?
        switch noteMatrixCandidate {
        case .nothing:
            break
        case .fixed:
            adjacencyForNotes = self.performSigmoidOn(matrix: self.textualSimilarityMatrix.matrix[0..<self.notes.count, 0..<self.textualSimilarityMatrix.matrix.cols] + self.entitiesMatrix.matrix[0..<self.notes.count, 0..<self.entitiesMatrix.matrix.cols], middle: 1.0, beta: self.beta)
        case .combinationAfterSigmoid:
            adjacencyForNotes = (self.weights[.text] ?? 0.5) .* self.performSigmoidOn(matrix: self.textualSimilarityMatrix.matrix[0..<self.notes.count, 0..<self.textualSimilarityMatrix.matrix.cols], middle: 1, beta: self.beta) + (self.weights[.entities] ?? 0.5) .* self.performSigmoidOn(matrix: self.entitiesMatrix.matrix[0..<self.notes.count, 0..<self.entitiesMatrix.matrix.cols], middle: 0.5, beta: beta)
        case .sigmoidOnEntities:
            adjacencyForNotes = self.performSigmoidOn(matrix: self.entitiesMatrix.matrix[0..<self.notes.count, 0..<self.entitiesMatrix.matrix.cols], middle: 0.3, beta: beta)
        case .combinationBeforeSigmoid:
            adjacencyForNotes = self.performSigmoidOn(matrix: (self.weights[.text] ?? 0.5) .* self.textualSimilarityMatrix.matrix[0..<self.notes.count, 0..<self.textualSimilarityMatrix.matrix.cols] + (self.weights[.entities] ?? 0.5) .* self.entitiesMatrix.matrix[0..<self.notes.count, 0..<self.entitiesMatrix.matrix.cols], middle: (self.weights[.navigation] ?? 0.5) * 2, beta: self.beta)
        }

        if let adjacencyForNotes = adjacencyForNotes {
            self.adjacencyMatrix[0..<self.notes.count, 0..<self.adjacencyMatrix.cols] = adjacencyForNotes
            self.adjacencyMatrix[0..<self.adjacencyMatrix.rows, 0..<self.notes.count] = adjacencyForNotes.T
        }

        // Add threshold for very small values
        self.adjacencyMatrix = self.threshold(matrix: self.adjacencyMatrix, threshold: 0.0001)
    }

    public func removeNote(noteId: UUID) {
        self.additionsInQueue += 1
        additionQueue.async {
            if let noteIndex = self.findNoteInNotes(noteID: noteId) {
                if self.adjacencyMatrix.rows > 1 {
                    do {
                        try self.navigationMatrix.removeDataPoint(index: noteIndex)
                        try self.textualSimilarityMatrix.removeDataPoint(index: noteIndex)
                        try self.entitiesMatrix.removeDataPoint(index: noteIndex)
                        try self.beTogetherMatrix.removeDataPoint(index: noteIndex)
                        try self.beApartMatrix.removeDataPoint(index: noteIndex)
                        self.createAdjacencyMatrix()
                    } catch { } // This is pretty bad
                }
                self.notes.remove(at: noteIndex)
            }
            self.mainQueue.async {
                self.additionsInQueue -= 1
            }
        }
    }
    
    /// Remove pages (only) from the adjacency matrix but keep them by fixing them
    /// to another page (only) which is the most similar at the specific moment
    ///
    /// - Parameters:
    ///   - ranking: A list of all pages, ranked in the order of their score (from
    ///             the lowest to the highest)
    func remove(ranking: [UUID], activeSources: [UUID]? = nil, numPagesToRemove: Int = 3) throws {
        var ranking = ranking
        var pagesRemoved = 0
        if let activeSources = activeSources {
            let rankingWithoutActive = ranking.filter { !activeSources.contains($0) }
            if rankingWithoutActive.count > 2 {
                ranking = rankingWithoutActive
            }
        }
        while pagesRemoved < numPagesToRemove && self.navigationMatrix.matrix.rows > 1 {
            if let pageToRemove = ranking.first {
                if let pageIndexToRemove = self.findPageInPages(pageID: pageToRemove) {
                    try self.navigationMatrix.removeDataPoint(index: pageIndexToRemove + self.notes.count)
                    try self.textualSimilarityMatrix.removeDataPoint(index: pageIndexToRemove + self.notes.count)
                    try self.entitiesMatrix.removeDataPoint(index: pageIndexToRemove + self.notes.count)
                    try self.beTogetherMatrix.removeDataPoint(index: pageIndexToRemove + self.notes.count)
                    try self.beApartMatrix.removeDataPoint(index: pageIndexToRemove + self.notes.count)
                    self.pages.remove(at: pageIndexToRemove)
                    pagesRemoved += 1
                    ranking = Array(ranking.dropFirst())
                } else { ranking = Array(ranking.dropFirst()) }
            } else { break }
            self.createAdjacencyMatrix()
        }
    }

    /// Preprocess title to facilitate finding of entities
    ///
    /// - Parameters:
    ///   - title: The title to be preprocesses
    /// - Returns: A preprocessed version of the title
    func titlePreprocessing(of title: String, adress url: URL? = nil) -> String {
        let hostTokens = url?.host?.components(separatedBy: CharacterSet(charactersIn: "./"))
        var preprocessedTitle = title.lowercased().applyingTransform(.stripDiacritics, reverse: false) ?? title.lowercased()
        for prefix in (self.titlePrefixes + (hostTokens ?? [])) {
            if preprocessedTitle.hasPrefix(prefix.lowercased()) {
                preprocessedTitle = String(preprocessedTitle.dropFirst(prefix.count))
            }
        }
        for suffix in (self.titleSuffixes + (hostTokens ?? [])) {
            if preprocessedTitle.hasSuffix(suffix.lowercased()) {
                preprocessedTitle = String(preprocessedTitle.dropLast(suffix.count))
            }
        }
        let charactersToTrim: CharacterSet = .whitespaces.union(.punctuationCharacters).union(CharacterSet(charactersIn: "|"))
        preprocessedTitle = preprocessedTitle.capitalized.trimmingCharacters(in: charactersToTrim) //+ " and some text"
        return preprocessedTitle
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
    func createSimilarities(pageGroups: [[UUID]], noteGroups: [[UUID]], activeSources: [UUID]?) -> [UUID: [UUID: Double]] {
        var similarities = [UUID: [UUID: Double]]()
        // Start by including similarities with notes
        for (pageGroup, noteGroup) in zip(pageGroups, noteGroups) {
            for noteId in noteGroup {
                if let noteIndex = self.findNoteInNotes(noteID: noteId) {
                    similarities[noteId] = [UUID: Double]()
                    for pageId in pageGroup {
                        if let pageIndex = self.findPageInPages(pageID: pageId) {
                            similarities[noteId]?[pageId] = self.entitiesMatrix.matrix[noteIndex, pageIndex + self.notes.count] + pow(self.textualSimilarityMatrix.matrix[noteIndex, pageIndex + self.notes.count], 4.0)
                        }
                    }
                }
            }
            if let activeSources = activeSources {
                for pageId in pageGroup {
                    if activeSources.contains(pageId),
                       let pageIndex = self.findPageInPages(pageID: pageId) {
                        similarities[pageId] = [UUID:Double]()
                        for suggestedPageId in pageGroup.filter({ $0 != pageId }) {
                            if let suggestedPageIndex = self.findPageInPages(pageID: suggestedPageId) {
                                similarities[pageId]?[suggestedPageId] = self.entitiesMatrix.matrix[pageIndex + self.notes.count, suggestedPageIndex + self.notes.count] + pow(self.textualSimilarityMatrix.matrix[pageIndex + self.notes.count, suggestedPageIndex + self.notes.count], 4.0)
                            }
                        }
                    }
                }
            }
        }
        return similarities
    }

    /// A simple function to make sure the matrices beTogetherMatrix and beApartMatrix follow in size the other matrices. Note that here the matrices only grow (beTogether as all 0s and beApart as all 1s except for the diagonal). Real information regarding beTogether and beApart is added elsewhere
    ///
    /// - Parameters:
    ///   - dataPointType: Is the data point being added a note or a page
    func addToBeWithAndBeApart(dataPointType: DataPoint) throws {
        if self.pages.count + self.notes.count > 1 {
            let scoresBeWith = [Double](repeating: 0.0, count: self.beTogetherMatrix.matrix.rows)
            let scoresBeApart = [Double](repeating: 1.0, count: self.beTogetherMatrix.matrix.rows)
            do {
                try self.beTogetherMatrix.addDataPoint(similarities: scoresBeWith, type: dataPointType, numExistingNotes: max(self.notes.count - 1, 0), numExistingPages: self.pages.count)
                try self.beApartMatrix.addDataPoint(similarities: scoresBeApart, type: dataPointType, numExistingNotes: max(self.notes.count - 1, 0), numExistingPages: self.pages.count)
            } catch let error {
                throw error
            }
        }
    }

    /// A function to add a data point to all sub matrices (navigation, textual and entity-based)
    ///
    /// - Parameters:
    ///   - page: The page to be added, in case the data point is a page
    ///   - note: The note to be added, in case the data point is a note
    ///   - replaceContent: A flag to declare that the request comes from PnS, the page
    ///                     already exists, and the text considered for the page should be replaced.
    func updateSubMatrices(page: Page? = nil, note: ClusteringNote? = nil, replaceContent: Bool = false) throws {
        // Decide if the received data point in a page or a note
        var dataPointType: DataPoint = .note
        if page != nil {
            dataPointType = .page
        }
        // Update page, if already exists
        if let page = page,
           let id_index = self.findPageInPages(pageID: page.id) {
            if replaceContent,
               let newContent = page.cleanedContent {
                // Update content through PnS
                let totalContentTokenized = (newContent + " " + (self.pages[id_index].cleanedContent ?? ""))
                self.pages[id_index].cleanedContent = totalContentTokenized
                //self.pages[id_index].language = self.extractor.getTextLanguage(text: totalContentTokenized)
                
                do {
                    try self.textualSimilarityProcess(index: id_index, dataPointType: dataPointType, changeContent: true)
                    try self.entitiesProcess(index: id_index, dataPointType: dataPointType, changeContent: true)
                } catch let error {
                    throw error
                }
            }
            if let myParent = page.parentId,
               let parent_index = self.findPageInPages(pageID: myParent) {
                // Page exists, new parenting relation
                self.navigationMatrix.matrix[id_index + self.notes.count, parent_index + self.notes.count] = 1.0
                self.navigationMatrix.matrix[parent_index + self.notes.count, id_index + self.notes.count] = 1.0
            }
            if let beWith = page.beWith {
                for parentId in beWith {
                    if let parent_index = self.findPageInPages(pageID: parentId),
                       parent_index != id_index {
                        self.beApartMatrix.matrix[id_index + self.notes.count, parent_index + self.notes.count] = 1.0
                        self.beApartMatrix.matrix[parent_index + self.notes.count, id_index + self.notes.count] = 1.0
                        self.beTogetherMatrix.matrix[id_index + self.notes.count, parent_index + self.notes.count] = 1.0
                        self.beTogetherMatrix.matrix[parent_index + self.notes.count, id_index + self.notes.count] = 1.0
                    }
                }
            }
            if let beApart = page.beApart {
                for parentId in beApart {
                    if let parent_index = self.findPageInPages(pageID: parentId),
                       parent_index != id_index {
                        self.beApartMatrix.matrix[id_index + self.notes.count, parent_index + self.notes.count] = 0.0
                        self.beApartMatrix.matrix[parent_index + self.notes.count, id_index + self.notes.count] = 0.0
                        self.beTogetherMatrix.matrix[id_index + self.notes.count, parent_index + self.notes.count] = 0.0
                        self.beTogetherMatrix.matrix[parent_index + self.notes.count, id_index + self.notes.count] = 0.0
                    }
                }
            }
            // Updating existing note
        } else if let note = note,
                  let id_index = self.findNoteInNotes(noteID: note.id) {
            do {
                if let newContent = note.originalContent {
                    //(self.notes[id_index].cleanedContent, self.notes[id_index].language) = try self.extractor.extract(from: newContent, forType: .note)
                    self.notes[id_index].cleanedContent = newContent.joined(separator: " ") //try self.extractor.extract(from: newContent, forType: .note)
                }
                if let newTitle = note.title {
                    self.notes[id_index].title = newTitle//self.titlePreprocessing(of: newTitle)
                }
                try self.textualSimilarityProcess(index: id_index, dataPointType: .note, changeContent: true)
                try self.entitiesProcess(index: id_index, dataPointType: .note, changeContent: true)
            } catch let error {
                throw error
            }
            // New page or note
        } else {
            // Navigation matrix computation
            var navigationSimilarities = [Double](repeating: 0.0, count: self.adjacencyMatrix.rows)
            
            var newIndex = self.pages.count
            if let page = page {
                self.pages.append(page)
                if let myParent = page.parentId, let parent_index = self.findPageInPages(pageID: myParent) {
                    navigationSimilarities[parent_index + self.notes.count] = 1.0
                }
                if let title = self.pages[newIndex].title {
                    self.pages[newIndex].title = title //self.titlePreprocessing(of: title, adress: page.url)
                }
                if page.cleanedContent == nil {
                    //do {
                        //(self.pages[newIndex].cleanedContent, self.pages[newIndex].language) = try self.extractor.extract(from: self.pages[newIndex].originalContent ?? [""])
                        self.pages[newIndex].cleanedContent = (self.pages[newIndex].originalContent ?? [""]).joined(separator: " ")
                        self.pages[newIndex].originalContent = nil
                    //} catch {
                    //}
                    //if self.pages[newIndex].cleanedContent?.count ?? 0 > 10 {
                        self.pagesWithContent += 1
                    //}
                }
                self.pages[newIndex].domain = page.url?.host
            } else if let note = note {
                // If the note does not contain enough text, abort
                /*guard let content = note.originalContent,
                      content.map({ $0.split(separator: " ").count }).reduce(0, +) > self.noteContentThreshold else {
                          throw AdditionError.notEnoughTextInNote
                      }*/
                newIndex = self.notes.count
                self.notes.append(note)
                if let title = self.notes[newIndex].title {
                    self.notes[newIndex].title = title //self.titlePreprocessing(of: title)
                }
                do {
                    //(self.notes[newIndex].cleanedContent, self.notes[newIndex].language) = try self.extractor.extract(from: self.notes[newIndex].originalContent ?? [""], forType: .note)
                    self.notes[newIndex].cleanedContent = (self.notes[newIndex].originalContent ?? [""]).joined(separator: " ")
                    self.notes[newIndex].originalContent = nil
                }/* catch {
                }*/
            }
            // Add to submatrices
            do {
                try self.navigationMatrix.addDataPoint(similarities: navigationSimilarities, type: dataPointType, numExistingNotes: self.notes.count - Int(truncating: NSNumber(value: dataPointType == .note)), numExistingPages: self.pages.count - Int(truncating: NSNumber(value: dataPointType == .page)))
                try self.textualSimilarityProcess(index: newIndex, dataPointType: dataPointType)
                try self.entitiesProcess(index: newIndex, dataPointType: dataPointType)
                try self.addToBeWithAndBeApart(dataPointType: dataPointType)
            } catch let error {
                throw error
            }
        }
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
    public func add(page: Page? = nil, note: ClusteringNote? = nil, ranking: [UUID]?, activeSources: [UUID]? = nil, replaceContent: Bool = false, completion: @escaping (Result<(pageGroups: [[UUID]], noteGroups: [[UUID]], flag: Flag, similarities: [UUID: [UUID: Double]]), Error>) -> Void) {
        self.additionsInQueue += 1
        additionQueue.async {
            // Check that we are adding exactly one object
            if page != nil && note != nil {
                self.mainQueue.async {
                    self.additionsInQueue -= 1
                    completion(.failure(AdditionError.moreThanOneObjectToAdd))
                }
                return
            }
            if page == nil && note == nil {
                self.mainQueue.async {
                    self.additionsInQueue -= 1
                    completion(.failure(AdditionError.noObjectsToAdd))
                }
                return
            }
            // Making sure addition doesn't happen during clustering
            guard self.isClustering == false else {
                if let page = page {
                    self.skippedPages.append(page)
                }
                if let note = note {
                    self.skippedNotes.append(note)
                }
                self.mainQueue.async {
                    self.additionsInQueue -= 1
                    completion(.failure(AdditionError.abortingAdditionDuringClustering))
                }
                return
            }
            // If ranking is received, remove pages
            if let ranking = ranking {
                do {
                    try self.remove(ranking: ranking, activeSources: activeSources)
                } catch let error {
                    self.mainQueue.async {
                        completion(.failure(error))
                    }
                }
            }
            // Updating all sub-matrices
            do {
                try self.updateSubMatrices(page: page, note: note, replaceContent: replaceContent)
                self.createAdjacencyMatrix()
            } catch {
                self.mainQueue.async {
                    self.additionsInQueue -= 1
                    completion(.failure(error))
                }
                return
            }
            if self.pagesWithContent == 3 {
                self.askForNotes = true
            }
            
            self.mainQueue.async {
                self.additionsInQueue -= 1
                if self.additionsInQueue == 0 {
                    self.isClustering = true
                    self.clusteringQueue.async {
                        let start = CFAbsoluteTimeGetCurrent()
                        var predictedClusters = zeros(1, self.adjacencyMatrix.rows).flat.map { Int($0) }
                        do {
                            predictedClusters = try self.spectralClustering()
                        } catch let error {
                            completion(.failure(error))
                            self.mainQueue.async {
                                self.isClustering = false
                            }
                            return
                        }
                        let clusteringTime = CFAbsoluteTimeGetCurrent() - start
                        let stablizedClusters = self.stabilize(predictedClusters)
                        let (resultPages, resultNotes) = self.clusterizeIDs(labels: stablizedClusters)
                        let similarities = self.createSimilarities(pageGroups: resultPages, noteGroups: resultNotes, activeSources: activeSources)
                        
                        self.mainQueue.async {
                            self.isClustering = false
                            if self.askForNotes {
                                completion(.success((pageGroups: resultPages, noteGroups: resultNotes, flag: .addNotes, similarities: similarities)))
                                self.askForNotes = false
                            } else if clusteringTime > self.timeToRemove {
                                completion(.success((pageGroups: resultPages, noteGroups: resultNotes, flag: .sendRanking, similarities: similarities)))
                            } else {
                                completion(.success((pageGroups: resultPages, noteGroups: resultNotes, flag: .none, similarities: similarities)))
                            }
                        }
                    }
                } else {
                    completion(.failure(AdditionError.skippingToNextAddition))
                }
            }
        }
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
        guard self.notes.count + self.pages.count > 0 else { return ([[UUID]](), [[UUID]]())}

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

    /// Perform a cnadidate change (this is where we can change candidates)
    func performCandidateChange() throws {
        switch self.candidate {
        case 1:
            self.laplacianCandidate = LaplacianCandidate.nonNormalizedLaplacian
            self.matrixCandidate = SimilarityMatrixCandidate.navigationMatrix
            self.noteMatrixCandidate = SimilarityForNotesCandidate.nothing
            self.numClustersCandidate = NumClusterComputationCandidate.threshold
        case 2:
            self.laplacianCandidate = LaplacianCandidate.randomWalkLaplacian
            self.matrixCandidate = SimilarityMatrixCandidate.combinationSigmoidWithTextErasure
            self.noteMatrixCandidate = SimilarityForNotesCandidate.sigmoidOnEntities
            self.numClustersCandidate = NumClusterComputationCandidate.biggestDistanceInPercentages
        case 3:
            self.laplacianCandidate = LaplacianCandidate.randomWalkLaplacian
            self.matrixCandidate = SimilarityMatrixCandidate.fixedPagesTestNotes
            self.noteMatrixCandidate = SimilarityForNotesCandidate.combinationBeforeSigmoid
            self.numClustersCandidate = NumClusterComputationCandidate.biggestDistanceInPercentages
        default:
            throw CandidateError.unknownCandidate
        }
    }

    /// A function to allow clusteringManager to change between candidates for testing
    /// Changing candidates implies new clustering
    ///
    /// - Parameters:
    ///   - candidate: The candidate to change to
    ///   - weightNavigation: New weight for the navigation component
    ///   - weightText: New weight for the text component
    ///   - weightEntities: New weight for the entities component
    /// - Returns: (through completion)
    ///             - pageGroups: Array of arrays of all pages clustered into groups
    ///             - noteGroups: Array of arrays of all notes clustered into groups, corresponding to the groups of pages
    ///             - sendRanking: A flag to ask the clusteringManager to send page ranking with the next 'add' request, for the purpose of removing some pages
    // swiftlint:disable:next large_tuple
    public func changeCandidate(to candidate: Int?, with weightNavigation: Double?, with weightText: Double?, with weightEntities: Double?, activeSources: [UUID]? = nil, completion: @escaping (Result<(pageGroups: [[UUID]], noteGroups: [[UUID]], flag: Flag, similarities: [UUID: [UUID: Double]]), Error>) -> Void) {
        self.additionsInQueue += 1
        self.additionQueue.async {
            // If ranking is received, remove pages
            self.candidate = candidate ?? self.candidate
            self.weights[.navigation] = weightNavigation ?? self.weights[.navigation]
            self.weights[.text] = weightText ?? self.weights[.text]
            self.weights[.entities] = weightEntities ?? self.weights[.entities]
            do {
                try self.performCandidateChange()
            } catch {
                self.mainQueue.async {
                    self.additionsInQueue -= 1
                    completion(.failure(CandidateError.unknownCandidate))
                }
                return
            }
            
            self.mainQueue.async {
                self.additionsInQueue -= 1
                if self.additionsInQueue == 0 {
                    self.clusteringQueue.async {
                        self.createAdjacencyMatrix()
                        var predictedClusters = zeros(1, self.adjacencyMatrix.rows).flat.map { Int($0) }
                        do {
                            predictedClusters = try self.spectralClustering()
                        } catch let error {
                            completion(.failure(error))
                            return
                        }
                        let stablizedClusters = self.stabilize(predictedClusters)
                        let (resultPages, resultNotes) = self.clusterizeIDs(labels: stablizedClusters)
                        let similarities = self.createSimilarities(pageGroups: resultPages, noteGroups: resultNotes, activeSources: activeSources)
                        
                        self.mainQueue.async {
                            completion(.success((pageGroups: resultPages, noteGroups: resultNotes, flag: .none, similarities: similarities)))
                        }
                    }
                } else {
                    completion(.failure(AdditionError.skippingToNextAddition))
                }
            }
        }
    }
    // swiftlint:disable:next file_length
}
