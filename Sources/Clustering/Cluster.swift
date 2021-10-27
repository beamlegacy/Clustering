import LASwift
import Foundation
import NaturalLanguage
import Accelerate

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
    }

    let myQueue = DispatchQueue(label: "clusteringQueue")
    var pages = [Page]()
    var notes = [ClusteringNote]()
    // As the adjacency matrix is never touched on its own, just through the sub matrices, it does not need add or remove methods.
    var adjacencyMatrix = Matrix([[0]])
    var navigationMatrix = NavigationMatrix()
    var textualSimilarityMatrix = SimilarityMatrix()
    var entitiesMatrix = SimilarityMatrix()
    let tagger = NLTagger(tagSchemes: [.nameType])
    let entityOptions: NLTagger.Options = [.omitPunctuation, .omitWhitespace, .joinNames]
    let entityTags: [NLTag] = [.personalName, .placeName, .organizationName]
    var timeToRemove: Double = 0.5 // If clustering takes more than this (in seconds) we start removing pages
    let titleSuffixes = [" - Google Search", " - YouTube"]
    let beta = 50.0
    var noteContentThreshold: Int
    let extractor = JusText()

    //Define which Laplacian to use
    var laplacianCandidate = LaplacianCandidate.randomWalkLaplacian
    // Define which similarity matrix to use
    var matrixCandidate = SimilarityMatrixCandidate.combinationSigmoidWithTextErasure
    var noteMatrixCandidate = SimilarityForNotesCandidate.fixed
    // Define which number of clusters computation to use
    var numClustersCandidate = NumClusterComputationCandidate.biggestDistanceInPercentages
    var candidate: Int
    var weights = [AllWeights: Double]()

    public init(candidate: Int = 2, weightNavigation: Double = 0.5, weightText: Double = 0.9, weightEntities: Double = 0.4, noteContentThreshold: Int = 100) {
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

            switch index {
            case 0:
                self.matrix = self.matrix ?? (.Drop(1), .Drop(1))
            case self.matrix.rows - 1:
                self.matrix = self.matrix ?? (.DropLast(1), .DropLast(1))
            default:
                let upLeft = self.matrix ?? (.Take(index), .Take(index))
                let upRight = self.matrix ?? (.Take(index), .TakeLast(self.matrix.rows - index - 1))
                let downLeft = self.matrix ?? (.TakeLast(self.matrix.rows - index - 1), .Take(index))
                let downRight = self.matrix ?? (.TakeLast(self.matrix.rows - index - 1), .TakeLast(self.matrix.rows - index - 1))
                let left = upLeft === downLeft
                let right = upRight === downRight
                self.matrix = left ||| right
            }
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

    /// Perform spectral clustering over a given adjacency matrix
    ///
    /// - Returns: An array of integers, corresponding to a grouping of all data points. The number given to each group is meaningless, only the grouping itself is meaningful
    // swiftlint:disable:next cyclomatic_complexity function_body_length
    func spectralClustering() throws -> [Int] {
        guard self.adjacencyMatrix.rows >= 2 else {
            return zeros(1, self.adjacencyMatrix.rows).flat.map { Int($0) }
        }

        let d = reduce(self.adjacencyMatrix, sum, .Row)
        let d1: [Double] = d.map { elem in
            if elem < 1e-5 { return elem } else { return 1 / elem }
        }
        let D = diag(d)
        let D1 = diag(d1)
        // This naming makes sense as D1 is 1/D
        let laplacianNn = D - self.adjacencyMatrix
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

        guard numClusters > 1 else {
            return zeros(1, self.adjacencyMatrix.rows).flat.map { Int($0) }
        }
        if eigenVcts.cols > numClusters {
            eigenVcts = eigenVcts ?? (.All, .Take(numClusters))
        }
        var points = [Vector]()
        for row in 0..<eigenVcts.rows {
            points.append(Vector(eigenVcts[row: row]))
        }
        let labels = [Int](0...numClusters - 1)
        var bestPredictedLabels: [Int]?
        var bestLabelsScore: Int?
        var predictedLabels: [Int] = []
        let kmeans = KMeans(labels: labels)
        for _ in 1...3 {
            predictedLabels = []
            kmeans.trainCenters(points, convergeDistance: 0.00001)
            for point in points {
                predictedLabels.append(kmeans.fit(point))
            }
            var newLabelScore = 0
            for label in predictedLabels[0..<self.notes.count] {
                let numPointsWithNote = predictedLabels.filter{ $0 == label }.count - 1
                if numPointsWithNote > 0 {
                    newLabelScore += predictedLabels.filter{ $0 == label }.count
                } else {
                    newLabelScore -= 5
                }
            }
            if bestLabelsScore == nil || newLabelScore < (bestLabelsScore ?? 1000) {
                bestLabelsScore = newLabelScore
                bestPredictedLabels = predictedLabels
            }
        }
        if let bestPredictedLabels = bestPredictedLabels {
            return bestPredictedLabels
        } else {
            return predictedLabels
        }
    }

    ///  Compute the embedding of the given piece of text if the language of the text is detectable
    ///  and if the OS is at least MacOS 11 and iOS 14.
    ///
    /// - Parameters:
    ///   - text: The text that will be turned into a contextual vector (embedding)
    /// - Returns: The embedding of the given piece of text as an optional and the dominating language of the text, as an optional.
    func textualEmbeddingComputationWithNLEmbedding(text: String, language: NLLanguage) -> [Double]? {
        if #available(iOS 14, macOS 11, *), language != NLLanguage.undetermined {
            if let sentenceEmbedding = NLEmbedding.sentenceEmbedding(for: language),
               let vector = sentenceEmbedding.vector(for: text) {
                    return vector
            }
        }
        return nil
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

        return dotProduct / (vec1Normed * vec2Normed)
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
    func scoreTextualSimilarity(textualEmbedding: [Double]?, language: NLLanguage, index: Int, dataPointType: DataPoint, changeContent: Bool = false) -> [Double] {
        var scores = [Double]()
         for note in notes.enumerated() {
            if dataPointType == . note {
                if !changeContent && note.offset == index { break }
                scores.append(-0.5)
            } else if let textualVectorID = note.element.textEmbedding,
                      let textLanguage = note.element.language,
                      let textualEmbedding = textualEmbedding,
                      textLanguage == language {
                scores.append(self.cosineSimilarity(vector1: textualVectorID, vector2: textualEmbedding))
            } else {
                scores.append(-0.5)
            }
        }

        for page in pages.enumerated() {
            // The textual vector might be empty, when the OS is not up to date
            // then the score will be 0.0
            if page.offset == index && dataPointType == .page {
                if changeContent {
                    scores.append(-0.5)
                } else { break }
            } else if let textLanguage = page.element.language,
                      textLanguage == language {
                if let textualVectorID = page.element.textEmbedding,
                   let textualEmbedding = textualEmbedding {
                    scores.append(self.cosineSimilarity(vector1: textualVectorID, vector2: textualEmbedding))
                } else {
                    scores.append(0.0)
                }
            } else if dataPointType == .page {
                scores.append(1.0) // We don't want to "break" connections between langauges
            } else {
                scores.append(0.0)
            }
        }
        return scores
    }

    /// Compute the entity similarity between the current data point to be added/changed and all existing data points
    ///
    /// - Parameters:
    ///   - entitiesInText: The entities found in the current text
    ///   - whichText: Is the current text the content or title of the data point
    ///   - index: The index of the data point within the corresponding vector (pages or notes)
    ///   - dataPointType: page or note
    ///   - changeContent: Is this a part of a content changing operation (rather than addition)
    /// - Returns: A list of similarity scores
    // swiftlint:disable:next cyclomatic_complexity
    func scoreEntitySimilarities(entitiesInNewText: EntitiesInText, in whichText: FindEntitiesIn, index: Int, dataPointType: DataPoint, changeContent: Bool = false) -> [Double] {
        var scores = [Double]()
        switch whichText {
        case .content:
            for note in notes.enumerated() {
                if dataPointType == .note {
                    if !changeContent && note.offset == index { break }
                    scores.append(0.0)
                } else if let entitiesInNote = note.element.entities {
                    scores.append(self.jaccardEntities(entitiesText1: entitiesInNewText, entitiesText2: entitiesInNote))
                } else {
                    scores.append(-0.5)
                }
            }
            for page in pages.enumerated() {
                if page.offset == index  && dataPointType == .page {
                    if changeContent {
                        scores.append(0.0)
                    } else { break }
                } else if let entitiesInPage = page.element.entities {
                    scores.append(self.jaccardEntities(entitiesText1: entitiesInNewText, entitiesText2: entitiesInPage))
                } else { scores.append(0.0) }
            }
        case .title:
            for note in notes.enumerated() {
                if dataPointType == .note {
                    if !changeContent && note.offset == index { break }
                    scores.append(0.0)
                } else if let entitiesInNoteTitle = note.element.entitiesInTitle {
                    scores.append(jaccardEntities(entitiesText1: entitiesInNewText, entitiesText2: entitiesInNoteTitle))
                } else {
                    scores.append(0.0)
                }
            }
            for page in pages.enumerated() {
                if page.offset == index && dataPointType == .page {
                    if changeContent {
                        scores.append(0.0)
                    } else { break }
                } else if let entitiesInPage = page.element.entitiesInTitle {
                    scores.append(self.jaccardEntities(entitiesText1: entitiesInNewText, entitiesText2: entitiesInPage))
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
        var language: NLLanguage?
        var scores = [Double](repeating: 0.0, count: self.textualSimilarityMatrix.matrix.rows)
        if dataPointType == .page {
            scores = [Double](repeating: 0.0, count: max(self.notes.count, 0))
            scores += [Double](repeating: 1.0, count: max(self.pages.count - 1, 0))
            if changeContent {
                scores.append(1.0)
            }
            content = pages[index].cleanedContent
            language = pages[index].language
        } else {
            scores = [Double](repeating: 0.0, count: max(self.notes.count - 1, 0))
            scores += [Double](repeating: 1.0, count: max(self.pages.count, 0))
            if changeContent {
                scores.insert(0.0, at: 0)
            }
            content = notes[index].content
            language = notes[index].language
        }
        if let content = content,
           let language = language {
            let textualEmbedding = self.textualEmbeddingComputationWithNLEmbedding(text: content, language: language)
            scores = self.scoreTextualSimilarity(textualEmbedding: textualEmbedding, language: language, index: index, dataPointType: dataPointType, changeContent: changeContent)
            switch dataPointType {
            case .page:
                pages[index].textEmbedding = textualEmbedding
            case .note:
                notes[index].textEmbedding = textualEmbedding
            }
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
        var scores = [Double](repeating: 0.0, count: self.entitiesMatrix.matrix.rows)
        var content: String?
        var title: String?
        switch dataPointType {
        case .page:
            content = self.pages[index].cleanedContent
            title = self.pages[index].title
        case .note:
            content = self.notes[index].content
            title = self.notes[index].title
        }
        if let content = content {
            let entitiesInNewText = self.findEntitiesInText(text: content)
            scores = zip(scores, self.scoreEntitySimilarities(entitiesInNewText: entitiesInNewText, in: FindEntitiesIn.content, index: index, dataPointType: dataPointType, changeContent: changeContent)).map({ $0.0 + $0.1 })
            switch dataPointType {
            case .page:
                pages[index].entities = entitiesInNewText
            case .note:
                notes[index].entities = entitiesInNewText
            }
        }
        if let title = title {
            let entitiesInNewTitle = findEntitiesInText(text: title)
            scores = zip(scores, self.scoreEntitySimilarities(entitiesInNewText: entitiesInNewTitle, in: FindEntitiesIn.title, index: index, dataPointType: dataPointType, changeContent: changeContent)).map({ min($0.0 + $0.1, 1.0) })
            switch dataPointType {
            case .page:
                pages[index].entitiesInTitle = entitiesInNewTitle
            case .note:
                notes[index].entitiesInTitle = entitiesInNewTitle
            }
        }

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

    /// Score the similarity between two EntitiesInText structs, using a modified Jaccard similarity
    ///
    /// - Parameters:
    ///   - entitiesInText1: The entities found in the first text
    ///   - entitiesInText1: The entities found in the first text
    /// - Returns: The similarity between the two data points, between 0 and 1
    func jaccardEntities(entitiesText1: EntitiesInText, entitiesText2: EntitiesInText) -> Double {
        var intersection: Set<String> = Set([String]())
        var union: Set<String> = Set([String]())
        var totalEntities1: Set<String> = Set([String]())
        var totalEntities2: Set<String> = Set([String]())

        for entityType in entitiesText1.entities.keys {
            union = Set(entitiesText1.entities[entityType] ?? [String]()).union(Set(entitiesText2.entities[entityType] ?? [String]())).union(union)
            intersection = Set(entitiesText1.entities[entityType] ?? [String]()).intersection(Set(entitiesText2.entities[entityType] ?? [String]())).union(intersection)
            totalEntities1 = Set(entitiesText1.entities[entityType] ?? [String]()).union(totalEntities1)
            totalEntities2 = Set(entitiesText2.entities[entityType] ?? [String]()).union(totalEntities2)
        }
        let minimumEntities = min(totalEntities1.count, totalEntities2.count)
        if minimumEntities > 0 {
            return Double(intersection.count) / Double(minimumEntities)
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
            let navigationSigmoidMatrix = self.performSigmoidOn(matrix: self.navigationMatrix.matrix, middle: self.weights[.navigation] ?? 0.5, beta: self.beta)
            let textSigmoidMatrix = self.performSigmoidOn(matrix: self.textualSimilarityMatrix.matrix, middle: self.weights[.text] ?? 0.5, beta: self.beta)
            let entitySigmoidMatrix = self.performSigmoidOn(matrix: self.entitiesMatrix.matrix, middle: self.weights[.entities] ?? 0.5, beta: self.beta)
            self.adjacencyMatrix = textSigmoidMatrix .* navigationSigmoidMatrix + entitySigmoidMatrix
        case .fixedPagesTestNotes:
            let navigationSigmoidMatrix = self.performSigmoidOn(matrix: self.navigationMatrix.matrix, middle: 0.5, beta: self.beta)
            let textSigmoidMatrix = self.performSigmoidOn(matrix: self.textualSimilarityMatrix.matrix, middle: 0.9, beta: self.beta)
            let entitySigmoidMatrix = self.performSigmoidOn(matrix: self.entitiesMatrix.matrix, middle: 0.4, beta: self.beta)
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
            adjacencyForNotes = self.performSigmoidOn(matrix: self.textualSimilarityMatrix.matrix[0..<self.notes.count, 0..<self.textualSimilarityMatrix.matrix.cols] + self.entitiesMatrix.matrix[0..<self.notes.count, 0..<self.entitiesMatrix.matrix.cols], middle: 1.2, beta: self.beta)
        case .combinationAfterSigmoid:
            adjacencyForNotes = (self.weights[.text] ?? 0.5) .* self.performSigmoidOn(matrix: self.textualSimilarityMatrix.matrix[0..<self.notes.count, 0..<self.textualSimilarityMatrix.matrix.cols], middle: 1, beta: self.beta) + (self.weights[.entities] ?? 0.5) .* self.performSigmoidOn(matrix: self.entitiesMatrix.matrix[0..<self.notes.count, 0..<self.entitiesMatrix.matrix.cols], middle: (self.weights[.navigation] ?? 0.5) * 2, beta: beta)
        case .combinationBeforeSigmoid:
            adjacencyForNotes = self.performSigmoidOn(matrix: (self.weights[.text] ?? 0.5) .* self.textualSimilarityMatrix.matrix[0..<self.notes.count, 0..<self.textualSimilarityMatrix.matrix.cols] + (self.weights[.entities] ?? 0.5) .* self.entitiesMatrix.matrix[0..<self.notes.count, 0..<self.entitiesMatrix.matrix.cols], middle: (self.weights[.navigation] ?? 0.5) * 2, beta: self.beta)
        }

        if let adjacencyForNotes = adjacencyForNotes {
            self.adjacencyMatrix[0..<self.notes.count, 0..<self.adjacencyMatrix.cols] = adjacencyForNotes
            self.adjacencyMatrix[0..<self.adjacencyMatrix.rows, 0..<self.notes.count] = adjacencyForNotes.T
        }
    }

    /// Remove pages (only) from the adjacency matrix but keep them by fixing them
    /// to another page (only) which is the most similar at the specific moment
    ///
    /// - Parameters:
    ///   - ranking: A list of all pages, ranked in the order of their score (from
    ///             the lowest to the highest)
    func remove(ranking: [UUID]) throws {
        var ranking = ranking
        var pagesRemoved = 0
        while pagesRemoved < 3 {
            if let pageToRemove = ranking.first {
                if let pageIndexToRemove = self.findPageInPages(pageID: pageToRemove) {
//                    var adjacencyVector = self.adjacencyMatrix[row: pageIndexToRemove + self.notes.count]
//                    adjacencyVector.removeFirst(self.notes.count)
//                    if let pageIndexToAttach = adjacencyVector.firstIndex(of: max(adjacencyVector)) {
//                        pages[pageIndexToAttach].attachedPages.append(pageToRemove)
//                        pages[pageIndexToAttach].attachedPages += pages[pageIndexToRemove].attachedPages
//                    }
                    try self.navigationMatrix.removeDataPoint(index: pageIndexToRemove + self.notes.count)
                    try self.textualSimilarityMatrix.removeDataPoint(index: pageIndexToRemove + self.notes.count)
                    try self.entitiesMatrix.removeDataPoint(index: pageIndexToRemove + self.notes.count)
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
    func titlePreprocessing(of title: String) -> String {
        var preprocessedTitle = title
        for suffix in self.titleSuffixes {
            if preprocessedTitle.hasSuffix(suffix) {
                preprocessedTitle = String(preprocessedTitle.dropLast(suffix.count))
                break
            }
        }
        preprocessedTitle = preprocessedTitle.capitalized + " and some text"
        return preprocessedTitle
    }

    /// When adding a new page, if it had been already visited and then deleted -
    /// remove from attachedPages of whatever page it was attached to
    ///
    /// - Parameters:
    ///   - newPageID: the ID of the page to add
    func removeFromDeleted(newPageID: UUID) {
        for page in self.pages.enumerated() {
            pages[page.offset].attachedPages = page.element.attachedPages.filter {$0 != newPageID}
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
    ///   - note: The note to be added, in case the data point is a note
    /// - Returns: (through completion)
    ///             - pageGroups: Array of arrays of all pages clustered into groups
    ///             - noteGroups: Array of arrays of all notes clustered into groups, corresponding to the groups of pages
    ///             - sendRanking: A flag to ask the clusteringManager to send page ranking with the next 'add' request, for the purpose of removing some pages
    // swiftlint:disable:next cyclomatic_complexity function_body_length large_tuple
    public func add(page: Page? = nil, note: ClusteringNote? = nil, ranking: [UUID]?, replaceContent: Bool = false, completion: @escaping (Result<(pageGroups: [[UUID]], noteGroups: [[UUID]], sendRanking: Bool), Error>) -> Void) {
        myQueue.async {
            // Check that we are adding exactly one object
            if page != nil && note != nil {
                completion(.failure(AdditionError.moreThanOneObjectToAdd))
                return
            }
            if page == nil && note == nil {
                completion(.failure(AdditionError.noObjectsToAdd))
                return
            }
            var dataPointType: DataPoint = .note
            if page != nil {
                dataPointType = .page
            }
            // If ranking is received, remove pages
            if let ranking = ranking {
                do {
                    try self.remove(ranking: ranking)
                } catch let error {
                    completion(.failure(error))
                }
            }
            // Updating the data point, if already existant
            if let page = page,
               let id_index = self.findPageInPages(pageID: page.id) {
                if replaceContent,
                   let newContent = page.cleanedContent {
                    // Update content through PnS
                    let totalContentTokenized = (newContent + " " + (self.pages[id_index].cleanedContent ?? "")).split(separator: " ")
                    if totalContentTokenized.count > 512 {
                        self.pages[id_index].cleanedContent = totalContentTokenized.dropLast(totalContentTokenized.count - 512).joined(separator: " ")
                    } else {
                        self.pages[id_index].cleanedContent = totalContentTokenized.joined(separator: " ")
                    }
                    do {
                        try self.textualSimilarityProcess(index: id_index, dataPointType: dataPointType, changeContent: true)
                        try self.entitiesProcess(index: id_index, dataPointType: dataPointType, changeContent: true)
                    } catch let error {
                        completion(.failure(error))
                    }
               }
               if let myParent = page.parentId,
               let parent_index = self.findPageInPages(pageID: myParent) {
                // Page exists, new parenting relation
                self.navigationMatrix.matrix[id_index + self.notes.count, parent_index + self.notes.count] = 1.0
                self.navigationMatrix.matrix[parent_index + self.notes.count, id_index + self.notes.count] = 1.0
               }
            // Updating existing note
            } else if let note = note,
                      let id_index = self.findNoteInNotes(noteID: note.id) {
                do {
                    if let newContent = note.content {
                        self.notes[id_index].content = newContent
                        self.notes[id_index].language = self.extractor.getTextLanguage(text: self.notes[id_index].content ?? "")
                    }
                    if let newTitle = note.title {
                        self.notes[id_index].title = self.titlePreprocessing(of: newTitle)
                    }
                    try self.textualSimilarityProcess(index: id_index, dataPointType: .note, changeContent: true)
                    try self.entitiesProcess(index: id_index, dataPointType: .note, changeContent: true)
                } catch let error {
                    completion(.failure(error))
                }
            // New page or note
            } else {
                // If page was visited in the past and deleted, remove from
                // deleted pages
                if let page = page {
                    self.removeFromDeleted(newPageID: page.id)
                }
                // If new note without enogh text, abort
                if let note = note {
                    guard let content = note.content,
                          content.split(separator: " ").count > self.noteContentThreshold else {
                              completion(.failure(AdditionError.notEnoughTextInNote))
                              return
                          }
                }
                // Navigation matrix computation
                var navigationSimilarities = [Double](repeating: 0.0, count: self.adjacencyMatrix.rows)

                if let page = page, let myParent = page.parentId, let parent_index = self.findPageInPages(pageID: myParent) {
                    navigationSimilarities[parent_index + self.notes.count] = 1.0
                }

                do {
                    try self.navigationMatrix.addDataPoint(similarities: navigationSimilarities, type: dataPointType, numExistingNotes: self.notes.count, numExistingPages: self.pages.count)
                } catch let error {
                    completion(.failure(error))
                }

                var newIndex = self.pages.count
                if let page = page {
                    self.pages.append(page)
                    if let title = self.pages[newIndex].title {
                        self.pages[newIndex].title = self.titlePreprocessing(of: title)
                    }
                    do {
                        (self.pages[newIndex].cleanedContent, self.pages[newIndex].language) = try self.extractor.extract(from: self.pages[newIndex].originalContent ?? [""])
                        self.pages[newIndex].originalContent = nil
                    } catch {
                    }
                } else if let note = note {
                    newIndex = self.notes.count
                    self.notes.append(note)
                    if let title = self.notes[newIndex].title {
                        self.notes[newIndex].title = self.titlePreprocessing(of: title)
                    }
                    self.notes[newIndex].language = self.extractor.getTextLanguage(text: self.notes[newIndex].content ?? "")
                }
                // Handle Text similarity and entities
                do {
                    try self.textualSimilarityProcess(index: newIndex, dataPointType: dataPointType)
                    try self.entitiesProcess(index: newIndex, dataPointType: dataPointType)
                } catch let error {
                    completion(.failure(error))
                }
            }
            //Here is where we would add more similarity matrices in the future
            self.createAdjacencyMatrix()

            let start = CFAbsoluteTimeGetCurrent()
            var predictedClusters = zeros(1, self.adjacencyMatrix.rows).flat.map { Int($0) }
            do {
                predictedClusters = try self.spectralClustering()
            } catch let error {
                completion(.failure(error))
            }
            let clusteringTime = CFAbsoluteTimeGetCurrent() - start
            let stablizedClusters = self.stabilize(predictedClusters)
            let (resultPages, resultNotes) = self.clusterizeIDs(labels: stablizedClusters)

            DispatchQueue.main.async {
                if clusteringTime > self.timeToRemove {
                    completion(.success((pageGroups: resultPages, noteGroups: resultNotes, sendRanking: true)))
                } else {
                    completion(.success((pageGroups: resultPages, noteGroups: resultNotes, sendRanking: false)))
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
                clusterizedPages[label.element] += self.pages[label.offset - self.notes.count].attachedPages
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
            self.noteMatrixCandidate = SimilarityForNotesCandidate.fixed
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
    public func changeCandidate(to candidate: Int?, with weightNavigation: Double?, with weightText: Double?, with weightEntities: Double?, completion: @escaping (Result<(pageGroups: [[UUID]], noteGroups: [[UUID]], sendRanking: Bool), Error>) -> Void) {
        myQueue.async {
            // If ranking is received, remove pages
            self.candidate = candidate ?? self.candidate
            self.weights[.navigation] = weightNavigation ?? self.weights[.navigation]
            self.weights[.text] = weightText ?? self.weights[.text]
            self.weights[.entities] = weightEntities ?? self.weights[.entities]
            do {
                try self.performCandidateChange()
            } catch {
                completion(.failure(CandidateError.unknownCandidate))
            }

            self.createAdjacencyMatrix()
            var predictedClusters = zeros(1, self.adjacencyMatrix.rows).flat.map { Int($0) }
            do {
                predictedClusters = try self.spectralClustering()
            } catch let error {
                completion(.failure(error))
            }
            let stablizedClusters = self.stabilize(predictedClusters)
            let (resultPages, resultNotes) = self.clusterizeIDs(labels: stablizedClusters)

            DispatchQueue.main.async {
                completion(.success((pageGroups: resultPages, noteGroups: resultNotes, sendRanking: false)))
            }
        }
    }
    // swiftlint:disable:next file_length
}
