import Foundation
import Accelerate
import CClustering


enum CClusteringError: Error {
    case tokenizerInitialization
}


class ModelInference {
    lazy var model: UnsafeMutableRawPointer! = {
        guard var modelPath = Bundle.module.path(forResource: "minilm_multilingual", ofType: "dylib"),
           var tokenizerModelPath = Bundle.module.path(forResource: "sentencepiece", ofType: "bpe.model")
        else {
          fatalError("Resources not found")
        }
        //var modelPath = "/Users/jplu/dev/clustering/Sources/Clustering/Resources/minilm_multilingual.dylib"
        //var tokenizerModelPath = "/Users/jplu/dev/clustering/Sources/Clustering/Resources/sentencepiece.bpe.model"
        var model: UnsafeMutableRawPointer!
        let bytesModel = modelPath.utf8CString
        let bytesTokenizer = tokenizerModelPath.utf8CString
        
        bytesModel.withUnsafeBufferPointer { ptrModel in
            bytesTokenizer.withUnsafeBufferPointer { ptrTokenizer in
                model = createModelInferenceWrapper(ptrModel.baseAddress, ptrTokenizer.baseAddress)
            }
        }
        /*modelPath.withUTF8 { cModelPath in
            tokenizerModelPath.withUTF8 { cTokenizerModelPath in
                model = createModelInferenceWrapper(cModelPath.baseAddress, cTokenizerModelPath.baseAddress)
            }
        }*/

        return model
    }()
    
    @MainActor func encode(text: String) async throws -> [Double] {
        //var content = text
        var result = ModelInferenceResult()
        var ret: Int32 = -1
        
        /*content.withUTF8 { cText in
            ret = doModelInference(self.model, cText.baseAddress, &result)
        }*/
        let bytesText = text.utf8CString
        
        bytesText.withUnsafeBufferPointer { ptrText in
            ret = doModelInference(self.model, ptrText.baseAddress, &result)
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


extension Double {
    /// Rounds the double to decimal places value
    func rounded(toPlaces places:Int) -> Double {
        let divisor = pow(10.0, Double(places))
        return (self * divisor).rounded() / divisor
    }
}


public class SmartClustering {
    var thresholdComparison = 0.3105
    var textualItems = [TextualItem]()
    var clusters = [[UUID]]()
    var similarities = [[Double]]()
    @MainActor let modelInf = ModelInference()

    public init() {}

    func normalize(vector: [Double]) -> [Double] {
        let normValue = cblas_dnrm2(Int32(vector.count), vector, 1)
        
        if normValue > 0 {
            var normalizedVector = [Double]()
            
            for i in 0...vector.count - 1 {
                normalizedVector.append((vector[i] / normValue).rounded(toPlaces: 4))
            }
            
            return normalizedVector
        }
        
        return vector
    }

    /// Compute the cosine similarity between two vectors
    ///
    /// - Parameters:
    ///   - vector1: a vector
    ///   - vector2: a vector
    /// - Returns: The cosine similarity between the two given vectors.
    func cosineSimilarity(vector1: [Double], vector2: [Double]) -> Double {
        let vector1Norm = normalize(vector: vector1)
        let vector2Norm = normalize(vector: vector2)
        
        if vector1Norm == [Double](repeating: 0.0, count: vector1.count) || vector2Norm == [Double](repeating: 0.0, count: vector1.count) {
            return 0.0.rounded(toPlaces: 4)
        }
        
        let vector1NormVector2NormDotProduct = cblas_ddot(Int32(vector1Norm.count), vector1Norm, 1, vector2Norm, 1)
        let vector1NormDotProduct = cblas_ddot(Int32(vector1Norm.count), vector1Norm, 1, vector1Norm, 1)
        let vector2NormDotProduct = cblas_ddot(Int32(vector2Norm.count), vector2Norm, 1, vector2Norm, 1)
        let sqrtVector1NormDotProduct = pow(vector1NormDotProduct, 0.5)
        let sqrtVector2NormDotProduct = pow(vector2NormDotProduct, 0.5)
        let similarity = vector1NormVector2NormDotProduct / (sqrtVector1NormDotProduct * sqrtVector2NormDotProduct)
        
        return similarity.rounded(toPlaces: 4)
    }

    func overallCosineSimilarity() -> [[Double]] {
        var cosineSimilarities = [[Double]]()
        
        for i in 0...self.textualItems.count - 1 {
            var currentCosineSimilarities = [Double]()
            
            for j in 0...self.textualItems.count - 1 {
                currentCosineSimilarities.append(self.cosineSimilarity(vector1: self.textualItems[i].embedding, vector2: self.textualItems[j].embedding))
            }
            
            cosineSimilarities.append(currentCosineSimilarities)
        }
        
        return cosineSimilarities
    }

    func argsort<T:Comparable>( a : [T] ) -> [Int] {
        var r = Array(a.indices)
        
        r.sort(by: { a[$0] > a[$1] })
        
        return r
    }

    func topk(k: Int, vector: [Double]) -> ([Double], [Int]) {
        var sortedVector = vector
        
        sortedVector.sort(by: { $0 > $1 })
        
        return (Array(sortedVector[0...k-1]), Array(self.argsort(a: vector)[0...k-1]))
    }

    func overallTopk(k: Int, vector: [[Double]]) -> ([[Double]], [[Int]]) {
        var values = [[Double]]()
        var indices = [[Int]]()
        
        for i in 0...vector.count - 1 {
            let tmpValuesIndices = self.topk(k: k, vector: vector[i])
            
            values.append(tmpValuesIndices.0)
            indices.append(tmpValuesIndices.1)
        }
        
        return (values, indices)
    }

    func createClusters() {
        var extractedClusters = [[Int]]()
        var nullClusters = [Int]()
        let sortMaxSize = self.textualItems.count
        let cosScores = self.overallCosineSimilarity()
        let topkValues = self.overallTopk(k: 1, vector: cosScores).0
        
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

    func findTextualItemIndex(of: UUID) -> Int {
        for (idx, val) in self.textualItems.enumerated() {
            if val.uuid == of {
                return idx
            }
        }
        
        return -1
    }
    
    func findTextualItemIndexInClusters(of: UUID) -> (Int, Int) {
        for (clusterIdx, cluster) in self.clusters.enumerated() {
            for (idx, uuid) in cluster.enumerated() {
                if uuid == of {
                    return (clusterIdx, idx)
                }
            }
        }
        
        return (-1, -1)
    }

    func createTextualItemGroups(of: TextualItemType) -> [[UUID]] {
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
    /// - Returns: - pageGroups: Array of arrays of all pages clustered into groups
    ///            - noteGroups: Array of arrays of all notes clustered into groups, corresponding to the groups of pages
    public func add(textualItem: TextualItem) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]]) {
        self.textualItems.append(textualItem)
        
        for (idx, item) in self.textualItems.enumerated() {
            if item.embedding.count == 0 {
                let text = (textualItem.title + " " + textualItem.content).trimmingCharacters(in: .whitespacesAndNewlines)
                
                if text.isEmpty {
                    self.textualItems[idx].updateEmbedding(newEmbedding: [Double](repeating: 0.0, count: 384))
                } else {
                    let embedding = try await self.modelInf.encode(text: text)
                    self.textualItems[idx].updateEmbedding(newEmbedding: embedding)
                }
            }
        }
        
        self.createClusters()
        
        let pageGroups = self.createTextualItemGroups(of: TextualItemType.page)
        let noteGroups = self.createTextualItemGroups(of: TextualItemType.note)
        
        return (pageGroups: pageGroups, noteGroups: noteGroups)
    }
    
    public func changeCandidate(threshold: Double) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]]) {
        self.thresholdComparison = threshold
        
        self.createClusters()
        
        let pageGroups = self.createTextualItemGroups(of: TextualItemType.page)
        let noteGroups = self.createTextualItemGroups(of: TextualItemType.note)
        
        return (pageGroups: pageGroups, noteGroups: noteGroups)
    }
}
