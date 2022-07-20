import Foundation
import Accelerate
import CClustering


enum CClusteringError: Error {
    case tokenizerError
    case ModelError
}


class ModelInference {
    let hidden_size: Int32 = 384
    lazy var model: UnsafeMutableRawPointer = {
        guard var modelPath = Bundle.module.path(forResource: "model-optimized-int32-quantized", ofType: "onnx", inDirectory: "Resources") else {
          fatalError("Resources not found")
        }
        
        var model: UnsafeMutableRawPointer!
        let bytesModel = modelPath.utf8CString
        //let bytesTokenizer = tokenizerModelPath.utf8CString
        
        bytesModel.withUnsafeBufferPointer { ptrModel in
            model = createModel(ptrModel.baseAddress, self.hidden_size)
        }
        // The comments below represents the way to do use UTF-8 C Strings with >= Swift 5.6.1. The day we will switch
        // to this version we could uncomment this part.
        /*modelPath.withUTF8 { cModelPath in
            model = createModel(ptrModel.baseAddress, 384)
        }*/

        return model
    }()
    
    lazy var tokenizer: UnsafeMutableRawPointer = {
        guard var tokenizerModelPath = Bundle.module.path(forResource: "sentencepiece", ofType: "bpe.model", inDirectory: "Resources")
        else {
          fatalError("Resources not found")
        }
        
        var tokenizer: UnsafeMutableRawPointer!
        let bytesTokenizer = tokenizerModelPath.utf8CString
        
        bytesTokenizer.withUnsafeBufferPointer { ptrTokenizer in
                tokenizer = createTokenizer(ptrTokenizer.baseAddress, 128)
        }
        // The comments below represents the way to do use UTF-8 C Strings with >= Swift 5.6.1. The day we will switch
        // to this version we could uncomment this part.
        /*tokenizerModelPath.withUTF8 { cTokenizerModelPath in
            tokenizer = createTokenizer(ptrTokenizer.baseAddress, 128)
        }
        }*/

        return tokenizer
    }()
    
    @MainActor func encode(tokenizerResult: inout TokenizerResult) async throws -> [Double] {
        var result = ModelResult()
        var ret: Int32 = -1
        
        ret = predict(self.model, &tokenizerResult, &result)
            
        if ret == 0 {
            let vector = Array(UnsafeBufferPointer(start: result.weigths, count: Int(result.size)))
            
            return vector.map{Double($0)}
        }
        
        throw CClusteringError.ModelError
    }
    
    @MainActor func tokenizeText(text: String) async throws -> TokenizerResult {
        // The comments below represents the way to do use UTF-8 C Strings with >= Swift 5.6.1. The day we will switch
        // to this version we could uncomment this part.
        //var content = text
        var result = TokenizerResult()
        var ret: Int32 = -1
        
        /*content.withUTF8 { cText in
            ret = tokenize(self.tokenizer, ptrText.baseAddress, &result)
        }*/
        let bytesText = text.utf8CString
        
        bytesText.withUnsafeBufferPointer { ptrText in
            ret = tokenize(self.tokenizer, ptrText.baseAddress, &result)
        }
            
        if ret == 0 {
            return result
        }
        
        throw CClusteringError.tokenizerError
    }
    
    deinit {
        removeModel(self.model)
        removeTokenizer(self.tokenizer)
    }
}


public class SmartClustering {
    var thresholdComparison = 0.3105
    var textualItems = [TextualItem]()
    var clusters = [[UUID]]()
    var similarities = [[Double]]()
    @MainActor let modelInf = ModelInference()

    public init() {}
    
    /// Compute the pair-wised cosine similarity matrix across all the textual items.
    ///
    /// - Returns:  The pair-wised cosine similarity matrix.
    func cosineSimilarityMatrix() -> [[Double]] {
        var cosineSimilarities = [[Double]]()
        
        for i in 0...self.textualItems.count - 1 {
            var currentCosineSimilarities = [Double]()
            
            for j in 0...self.textualItems.count - 1 {
                currentCosineSimilarities.append(MathsUtils.cosineSimilarity(vector1: self.textualItems[i].embedding, vector2: self.textualItems[j].embedding))
            }
            
            cosineSimilarities.append(currentCosineSimilarities)
        }
        
        return cosineSimilarities
    }

    /// Compute the top K values and indices.
    ///
    /// - Parameters:
    ///    - k: Size limit.
    ///    - vector: Vector from which to compute the top k.
    /// - Returns: - values: The top K array.
    ///            - indices: The indices of the top K array.
    private func topk(k: Int, vector: [Double]) -> (values: [Double], indices: [Int]) {
        var sortedVector = vector
        var indicesVector = Array(vector.indices)
        
        sortedVector.sort(by: { $0 > $1 })
        indicesVector.sort(by: { vector[$0] > vector[$1] })
        
        return (Array(sortedVector[0...k-1]), Array(indicesVector[0...k-1]))
    }

    /// Compute the pair-wised top K matrix values and indices from the pair-wised cosine similarity matrix.
    ///
    /// - Parameters:
    ///    - k: Size limit.
    ///    - vector: Vector from which to compute the top k.
    /// - Returns: - values: The pairwised top K matrix.
    ///            - indices: The indices of the pairwised top K matrix.
    private func topkMatrix(k: Int, vector: [[Double]]) -> (values: [[Double]], indices: [[Int]]) {
        var values = [[Double]]()
        var indices = [[Int]]()
        
        for i in 0...vector.count - 1 {
            let tmpValuesIndices = self.topk(k: k, vector: vector[i])
            
            values.append(tmpValuesIndices.0)
            indices.append(tmpValuesIndices.1)
        }
        
        return (values, indices)
    }

    /// The main function that creates the clusters.
    private func createClusters() {
        var extractedClusters = [[Int]]()
        var nullClusters = [Int]()
        let sortMaxSize = self.textualItems.count
        let cosScores = self.cosineSimilarityMatrix()
        let topkValues = self.topkMatrix(k: 1, vector: cosScores).0
        
        for i in 0...topkValues.count - 1 {
            if let lastElement = topkValues[i].last {
                if lastElement == 0.0 {
                    nullClusters.append(i)
                } else if (lastElement >= self.thresholdComparison) {
                    var newCluster = [Int]()
                    let topkRes = self.topk(k: sortMaxSize, vector: cosScores[i])
                    let topValLarge = topkRes.0
                    let topIdxLarge = topkRes.1
                    
                    if let lastVal = topValLarge.last {
                        if lastVal < self.thresholdComparison {
                            for (idx, val) in zip(topIdxLarge, topValLarge) {
                                if val < self.thresholdComparison {
                                    break
                                }
                                
                                newCluster.append(idx)
                            }
                        } else {
                            for (idx, val) in cosScores[i].enumerated() {
                                if val >= self.thresholdComparison {
                                    newCluster.append(idx)
                                }
                            }
                        }
                    }
                    
                    extractedClusters.append(newCluster)
                }
            }
        }
        
        if nullClusters.count > 0 {
            extractedClusters.append(nullClusters)
        }
        
        extractedClusters.sort(by: { $0.count > $1.count })
        
        self.clusters = []
        
        var extractedIds = Set<Int>()
        var total = 0
        
        for cluster in extractedClusters {
            var sortedCluster = cluster
            var nonOverlappedCluster = [UUID]()
            
            sortedCluster.sort(by: { $0 < $1 })
            
            for idx in sortedCluster {
                if !extractedIds.contains(idx) {
                    nonOverlappedCluster.append(self.textualItems[idx].uuid)
                    extractedIds.update(with: idx)
                }
            }
            
            if nonOverlappedCluster.count >= 1 {
                self.clusters.append(nonOverlappedCluster)
                total += nonOverlappedCluster.count
            }
        }
        
        assert(total == self.textualItems.count)
    }

    /// Find the index of a given UUID textual item.
    ///
    /// - Parameters:
    ///   - of: The textual item UUID to find.
    /// - Returns: The corresponding index.
    private func findTextualItemIndex(of: UUID) -> Int {
        for (idx, val) in self.textualItems.enumerated() {
            if val.uuid == of {
                return idx
            }
        }
        
        return -1
    }
    
    /// Find the cluster index of a given UUID textual item.
    ///
    /// - Parameters:
    ///   - of: The textual item UUID to find.
    /// - Returns: - clusterIndex: First dimension index.
    ///            - indexInCluster: Second dimension index.
    private func findTextualItemIndexInClusters(of: UUID) -> (clusterIndex: Int, indexInCluster: Int) {
        for (clusterIdx, cluster) in self.clusters.enumerated() {
            for (idx, uuid) in cluster.enumerated() {
                if uuid == of {
                    return (clusterIdx, idx)
                }
            }
        }
        
        return (-1, -1)
    }

    /// Create the clusters of a given textual item type.
    ///
    /// - Parameters:
    ///   - of: The type of the needed clusters.
    /// - Returns: The clusters.
    private func createTextualItemGroups(of: TextualItemType) -> [[UUID]] {
        var textualItemGroups = [[UUID]]()
        
        for cluster in self.clusters {
            var uniqueCluster = [UUID]()
            
            for val in cluster {
                let type = self.textualItems[self.findTextualItemIndex(of: val)].type
                
                if type == of {
                    uniqueCluster.append(val)
                }
            }
            
            textualItemGroups.append(uniqueCluster)
        }
        
        return textualItemGroups
    }

    /// Remove the given textual item and recompute the clusters.
    ///
    /// - Parameters:
    ///   - textualItem: The textual item to be removed.
    /// - Returns: - pageGroups: Newly computed pages cluster.
    ///            - noteGroups: Newly computed notes cluster.
    public func removeTextualItem(textualItemUUID: UUID) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]]) {
        let index = self.findTextualItemIndex(of: textualItemUUID)
        
        if index != -1 {
            self.textualItems.remove(at: index)
            let (clusterIdx, textualItemIdx) = self.findTextualItemIndexInClusters(of: textualItemUUID)
            
            if (clusterIdx != -1 && textualItemIdx != -1) {
                self.clusters[clusterIdx].remove(at: textualItemIdx)
            }
        }
        
        let pageGroups = self.createTextualItemGroups(of: TextualItemType.page)
        let noteGroups = self.createTextualItemGroups(of: TextualItemType.note)
        
        return (pageGroups: pageGroups, noteGroups: noteGroups)
    }

    /// The main function to access the package, adding a textual item
    /// to the clustering process.
    ///
    /// - Parameters:
    ///   - textualItem: The textual item to be added.
    /// - Returns: - pageGroups: Array of arrays of all pages clustered into groups.
    ///            - noteGroups: Array of arrays of all notes clustered into groups, corresponding to the groups of pages.
    public func add(textualItem: TextualItem) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]]) {
        self.textualItems.append(textualItem)
        
        for (idx, item) in self.textualItems.enumerated() {
            if item.embedding.count == 0 {
                let text = (textualItem.title + "</s></s>" + textualItem.content).trimmingCharacters(in: .whitespacesAndNewlines)
                
                if text.isEmpty {
                    self.textualItems[idx].updateEmbedding(newEmbedding: [Double](repeating: 0.0, count: Int(self.modelInf.hidden_size)))
                } else {
                    var tokenizedText = try await self.modelInf.tokenizeText(text: text)
                    let embedding = try await self.modelInf.encode(tokenizerResult: &tokenizedText)
                    self.textualItems[idx].updateEmbedding(newEmbedding: embedding)
                }
            }
        }
        
        self.createClusters()
        
        let pageGroups = self.createTextualItemGroups(of: TextualItemType.page)
        let noteGroups = self.createTextualItemGroups(of: TextualItemType.note)
        
        return (pageGroups: pageGroups, noteGroups: noteGroups)
    }

    /// Update the comparison threshold and recompute the clusters.
    ///
    /// - Parameters:
    ///   - threshold: The new comparison threshold.
    /// - Returns: - pageGroups: Newly computed pages cluster.
    ///            - noteGroups: Newly computed notes cluster.
    public func changeCandidate(threshold: Double) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]]) {
        self.thresholdComparison = threshold
        
        self.createClusters()
        
        let pageGroups = self.createTextualItemGroups(of: TextualItemType.page)
        let noteGroups = self.createTextualItemGroups(of: TextualItemType.note)
        
        return (pageGroups: pageGroups, noteGroups: noteGroups)
    }
}
