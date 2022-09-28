import Foundation
import Accelerate
import CClustering


enum CClusteringError: Error {
    case clusteringError
}


public class SmartClustering {
    var textualItems = [(UUID, UUID)]()
    let queue = DispatchQueue(label: "Clustering")
    var clustering: UnsafeMutableRawPointer!
    let websitesToUseOnlyTitle = ["youtube"]

    public init(threshold: Float = -1.0) {
        guard let modelPath = Bundle.module.path(forResource: "model-optimized-int32-quantized", ofType: "onnx", inDirectory: "Resources") else {
          fatalError("Resources not found")
        }
        guard let tokenizerModelPath = Bundle.module.path(forResource: "sentencepiece", ofType: "bpe.model", inDirectory: "Resources")
        else {
          fatalError("Resources not found")
        }
        let bytesModel = modelPath.utf8CString
        let bytesTokenizer = tokenizerModelPath.utf8CString
        
        bytesModel.withUnsafeBufferPointer { ptrModel in
            bytesTokenizer.withUnsafeBufferPointer { ptrTokenizer in
                self.clustering = createClustering(threshold, ptrModel.baseAddress, 384, ptrTokenizer.baseAddress, 128)
            }
        }
    }
    
    public func getThreshold() -> Float {
        return get_threshold(self.clustering)
    }

    /// Find the index of a given UUID textual item.
    ///
    /// - Parameters:
    ///   - of: The textual item UUID to find.
    ///   - from: The tab UUID containing the textual item
    /// - Returns: The corresponding index.
    private func findTextualItemIndex(of: UUID, from: UUID) -> Int {
        for (idx, textualItem) in self.textualItems.enumerated() {
            if textualItem.0 == of && from == textualItem.1  {
                return idx
            }
        }

        return -1
    }
    
    /// Format the C++ clustering result
    ///
    /// - Parameters:
    ///     - result: The clustering result from the C++ code base.
    /// - Returns: Formated clusters for Beam.
    private func formatClusteringResult(result: ClusteringResult) -> [[UUID]] {
        var clusters = [[UUID]]()
        let indices = Array(UnsafeBufferPointer(start: result.cluster.pointee.indices, count: Int(result.cluster.pointee.indices_size)))
        let clusters_split = Array(UnsafeBufferPointer(start: result.cluster.pointee.clusters_split, count: Int(result.cluster.pointee.clusters_split_size)))
        var start = 0
        
        for split in clusters_split {
            var cluster = [UUID]()
            
            for idx in start...(start + Int(split)) - 1 {
                cluster.append(self.textualItems[Int(indices[Int(idx)])].0)
            }

            start += Int(split)

            clusters.append(cluster)
        }
        
        return clusters
    }
    
    /// Remove the given textual item and recompute the clusters.
    ///
    /// - Parameters:
    ///   - textualItemIndex: The textual item to be removed.
    ///   - fromAdd: If the remove request comes from the add method or not.
    /// - Returns: The clustering result from the C++ code base.
    private func removeActualTextualItem(textualItemIndex: Int, fromAdd: Int32) throws -> ClusteringResult {
        self.textualItems.remove(at: textualItemIndex)
        
        var result = ClusteringResult()
        var ret: Int32 = -1
        
        ret = remove_textual_item(self.clustering, Int32(textualItemIndex), fromAdd, &result)
        
        if ret > 0 {
            throw CClusteringError.clusteringError
        }
        
        return result
    }

    /// Remove the given textual item and recompute the clusters.
    ///
    /// - Parameters:
    ///   - textualItem: The textual item to be removed.
    /// - Returns: Formated clusters for Beam.
    public func remove(textualItemUUID: UUID, textualItemTabId: UUID) async throws -> [[UUID]] {
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[[UUID]], Error>) in
            self.queue.async {
                do {
                    let idx = self.findTextualItemIndex(of: textualItemUUID, from: textualItemTabId)
                    var clusters = [[UUID]]()
                    
                    if idx != -1 {
                        let result = try self.removeActualTextualItem(textualItemIndex: idx, fromAdd: 0)
                        
                        clusters = self.formatClusteringResult(result: result)
                    }
            
                    continuation.resume(returning: clusters)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// The main function to access the package, adding a textual item
    /// to the clustering process.
    ///
    /// - Parameters:
    ///   - textualItem: The textual item to be added.
    /// - Returns: Formated clusters for Beam.
    public func add(textualItem: TextualItem) async throws -> [[UUID]] {
        repeat {
        } while self.clustering == nil
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[[UUID]], Error>) in
            self.queue.async {
                do {
                    var idx = self.findTextualItemIndex(of: textualItem.uuid, from: textualItem.tabId)
                    
                    if idx != -1 {
                        _ = try self.removeActualTextualItem(textualItemIndex: idx, fromAdd: 1)
                        
                        self.textualItems.insert((textualItem.uuid, textualItem.tabId), at: idx)
                    } else {
                        self.textualItems.append((textualItem.uuid, textualItem.tabId))
                        idx = self.textualItems.count - 1
                    }
                    
                    var text = ""
                    if !textualItem.url.isEmpty {
                        let comps = URLComponents(url: URL(string: textualItem.url)!, resolvingAgainstBaseURL: false)
                        
                        for website in self.websitesToUseOnlyTitle {
                            if let comps = comps {
                                if let host = comps.host {
                                    if host.contains(website) {
                                        text = (textualItem.processTitle() + "</s></s>").trimmingCharacters(in: .whitespacesAndNewlines)
                                    }
                                }
                            }
                        }
                    }
        
                    if text.isEmpty || text == "</s></s>" {
                        text = (textualItem.processTitle() + "</s></s>" + textualItem.content).trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    
                    var result = ClusteringResult()
                    var ret: Int32 = -1
                    let bytesContent = text.utf8CString
                    
                    bytesContent.withUnsafeBufferPointer { ptrContent in
                        ret = add_textual_item(self.clustering, ptrContent.baseAddress, Int32(idx), &result)
                    }
                    
                    if ret > 0 {
                        throw CClusteringError.clusteringError
                    }
                    
                    let clusters = self.formatClusteringResult(result: result)
                            
                    continuation.resume(returning: clusters)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Find a better threshold and recompute the clusters.
    ///
    /// - Parameters:
    ///      - expectedClusters: The gold representation of the clusters as expected
    /// - Returns: Formated clusters for Beam.
    public func changeCandidate(expectedClusters: [[TextualItem]]) async throws -> [[UUID]] {
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[[UUID]], Error>) in
            self.queue.async {
                do {
                    var result = ClusteringResult()
                    var ret: Int32 = -1
                    var expectedClustersDefinition = ClusterDefinition()
                    var indices = [UInt16]()
                    var clusters_split = [UInt16]()
                    
                    for cluster in expectedClusters {
                        clusters_split.append(UInt16(cluster.count))
                        
                        for textualItem in cluster {
                            indices.append(UInt16(self.findTextualItemIndex(of: textualItem.uuid, from: textualItem.tabId)))
                        }
                    }
                    
                    let pointer_indices = UnsafeMutablePointer<UInt16>.allocate(capacity: indices.count)
                    let pointer_clusters_split = UnsafeMutablePointer<UInt16>.allocate(capacity: clusters_split.count)
                    
                    pointer_indices.initialize(from: indices, count: indices.count)
                    pointer_clusters_split.initialize(from: clusters_split, count: clusters_split.count)
                    
                    expectedClustersDefinition.indices = pointer_indices
                    expectedClustersDefinition.indices_size = UInt16(indices.count)
                    expectedClustersDefinition.clusters_split = pointer_clusters_split
                    expectedClustersDefinition.clusters_split_size = UInt16(clusters_split.count)
                    
                    ret = recompute_clustering_threshold(self.clustering, &expectedClustersDefinition, &result)
                    
                    if ret > 0 {
                        throw CClusteringError.clusteringError
                    }
                    
                    let clusters = self.formatClusteringResult(result: result)
        
                    continuation.resume(returning: clusters)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
