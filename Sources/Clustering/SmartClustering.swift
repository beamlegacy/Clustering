import Foundation
import Accelerate
import CClustering


enum CClusteringError: Error {
    case clusteringError
}


public class SmartClustering {
    var textualItems = [TextualItem]()
    var pagesClusters = [[UUID]]()
    var notesClusters = [[UUID]]()
    var similarities = [[Float]]()
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
            if textualItem.uuid == of && from == textualItem.tabId  {
                return idx
            }
        }

        return -1
    }
    
    /// Find the cluster index of a given UUID textual item.
    ///
    /// - Parameters:
    ///   - of: The textual item UUID to find.
    ///   - from: The tab UUID containing the textual item
    /// - Returns: - clusterIndex: First dimension index.
    ///            - indexInCluster: Second dimension index.
    private func findTextualItemIndexInClusters(of: UUID, from: UUID) -> (clusterIndex: Int, indexInCluster: Int) {
        for (clusterIndex, cluster) in self.pagesClusters.enumerated() {
            for (indexInCluster, uuid) in cluster.enumerated() {
                if uuid == of {
                    let idx = self.findTextualItemIndex(of: uuid, from: from)
                    
                    if idx != -1 {
                        return (clusterIndex, indexInCluster)
                    }
                }
            }
        }
        
        for (clusterIndex, cluster) in self.notesClusters.enumerated() {
            for (indexInCluster, uuid) in cluster.enumerated() {
                if uuid == of {
                    let idx = self.findTextualItemIndex(of: uuid, from: from)
                    
                    if idx != -1 {
                        return (clusterIndex, indexInCluster)
                    }
                }
            }
        }
        
        return (-1, -1)
    }
    
    /// Remove the given textual item and recompute the clusters.
    ///
    /// - Parameters:
    ///   - textualItem: The textual item to be removed.
    /// - Returns: - pageGroups: Newly computed pages cluster.
    ///            - noteGroups: Newly computed notes cluster.
    private func removeActualTextualItem(textualItemIndex: Int, textualItemTabId: UUID) throws {
        let coordinates = self.findTextualItemIndexInClusters(of: self.textualItems[textualItemIndex].uuid, from: textualItemTabId)
        let uuidToRemove = self.textualItems[textualItemIndex].uuid
        let type = self.textualItems[textualItemIndex].type
        var ret: Int32 = -1
        
        ret = remove_textual_item(self.clustering, Int32(textualItemIndex))
        
        if ret > 0 {
            throw CClusteringError.clusteringError
        }
        
        self.textualItems.remove(at: textualItemIndex)
        
        if coordinates != (-1, -1) {
            if type == TextualItemType.page {
                self.pagesClusters[coordinates.clusterIndex].remove(at: coordinates.indexInCluster)
            } else {
                self.notesClusters[coordinates.clusterIndex].remove(at: coordinates.indexInCluster)
            }
        }
        
        for i in 0...self.similarities.count - 1 {
            self.similarities[i].remove(at: textualItemIndex)
        }

        self.similarities.remove(at: textualItemIndex)
        
        #if DEBUG
        print("FROM CLUSTERING - REMOVE - REMAINING PAGES AFTER REMOVING: ", uuidToRemove.description, " FROM Tab ID: ", textualItemTabId.description)
        for val in self.textualItems {
            print("FROM CLUSTERING - REMOVE - UUID: ", val.uuid)
            print("FROM CLUSTERING - REMOVE - TABID: ", val.tabId)
            print("FROM CLUSTERING - REMOVE - URL: ", val.url)
            print("FROM CLUSTERING - REMOVE - Title: ", val.title)
            print("FROM CLUSTERING - REMOVE - Processed Title: ", val.processTitle())
            print("FROM CLUSTERING - REMOVE - Content: ", val.content[val.content.startIndex..<String.Index(utf16Offset:min(val.content.count, 100), in: val.content)])
            print("--------")
        }
        print("FROM CLUSTERING - REMOVE - Similarities: ", self.similarities)
        #endif
    }

    /// Remove the given textual item and recompute the clusters.
    ///
    /// - Parameters:
    ///   - textualItem: The textual item to be removed.
    /// - Returns: - pageGroups: Newly computed pages cluster.
    ///            - noteGroups: Newly computed notes cluster.
    public func removeTextualItem(textualItemUUID: UUID, textualItemTabId: UUID) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]], similarities: [UUID: [UUID: Float]]) {
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<([[UUID]], [[UUID]], [UUID: [UUID: Float]]), Error>) in
            self.queue.async {
                do {
                    #if DEBUG
                    print("FROM CLUSTERING - REMOVE - REMOVING PAGE: ", textualItemUUID.description, " FROM Tab ID: ", textualItemTabId.description)
                    #endif
                    
                    let idx = self.findTextualItemIndex(of: textualItemUUID, from: textualItemTabId)
                    
                    if idx != -1 {
                        try self.removeActualTextualItem(textualItemIndex: idx, textualItemTabId: textualItemTabId)
                    } else {
                        #if DEBUG
                        print("FROM CLUSTERING - REMOVE - NOT FOUND PAGE: ", textualItemUUID.description, " FROM Tab ID: ", textualItemTabId.description)
                        #endif
                    }
                    
                    var similarities = [UUID: [UUID: Float]]()
                    
                    if self.textualItems.count > 0 {
                        similarities = self.createSimilarities()
                    }
            
                    continuation.resume(returning: (pageGroups: self.pagesClusters, noteGroups: self.notesClusters, similarities: similarities))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Turns the similarities matrix to a dict of dict.
    ///
    /// - Returns: A dict of dict representing the similarities across the textual items.
    private func createSimilarities() -> [UUID: [UUID: Float]] {
        var dict: [UUID: [UUID: Float]] = [:]
        
        for i in 0...self.textualItems.count - 1 {
            dict[self.textualItems[i].uuid] = [:]
            
            for j in 0...self.textualItems.count - 1 {
                dict[self.textualItems[i].uuid]?[self.textualItems[j].uuid] = self.similarities[i][j]
            }
        }
        
        return dict
    }

    /// The main function to access the package, adding a textual item
    /// to the clustering process.
    ///
    /// - Parameters:
    ///   - textualItem: The textual item to be added.
    /// - Returns: - pageGroups: Array of arrays of all pages clustered into groups.
    ///            - noteGroups: Array of arrays of all notes clustered into groups, corresponding to the groups of pages.
    ///            - similarities: Dict of dict of similiarity scores across each textual items.
    public func add(textualItem: TextualItem) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]], similarities: [UUID: [UUID: Float]]) {
        repeat {
        } while self.clustering == nil
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<([[UUID]], [[UUID]], [UUID: [UUID: Float]]), Error>) in
            self.queue.async {
                do {
                    let startTime = DispatchTime.now()
                    #if DEBUG
                    print("FROM CLUSTERING - ADD - ADDING PAGE: ", textualItem.uuid.description, " FROM Tab ID: ", textualItem.tabId.description)
                    #endif
                    
                    var idx = self.findTextualItemIndex(of: textualItem.uuid, from: textualItem.tabId)
                    
                    if idx != -1 {
                        #if DEBUG
                        print("FROM CLUSTERING - ADD - UUID: ", textualItem.uuid.description, " FROM Tab ID: ", textualItem.tabId.description, " already exists - delete first")
                        #endif
                        
                        _ = try self.removeActualTextualItem(textualItemIndex: idx, textualItemTabId: textualItem.tabId)
                        
                        self.textualItems.insert(textualItem, at: idx)
                    } else {
                        self.textualItems.append(textualItem)
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
                        ret = add_textual_item(self.clustering, ptrContent.baseAddress, &result)
                    }
                    
                    if ret > 0 {
                        throw CClusteringError.clusteringError
                    }
                    
                    let indices = Array(UnsafeBufferPointer(start: result.cluster.pointee.indices, count: Int(result.cluster.pointee.indices_size)))
                    let clusters_split = Array(UnsafeBufferPointer(start: result.cluster.pointee.clusters_split, count: Int(result.cluster.pointee.clusters_split_size)))
                    let sims = Array(UnsafeBufferPointer(start: result.similarities, count: self.textualItems.count * self.textualItems.count))
                    var start = 0
                    
                    self.pagesClusters.removeAll()
                    self.notesClusters.removeAll()
                    
                    for split in clusters_split {
                        var clusterPage = [UUID]()
                        var clusterNote = [UUID]()
                        
                        for idx in start...(start + Int(split)) - 1 {
                            if self.textualItems[Int(indices[Int(idx)])].type == .page {
                                clusterPage.append(self.textualItems[Int(indices[Int(idx)])].uuid)
                            } else {
                                clusterNote.append(self.textualItems[Int(indices[Int(idx)])].uuid)
                            }
                        }

                        start += Int(split)

                        self.pagesClusters.append(clusterPage)
                        self.notesClusters.append(clusterNote)
                    }
                    
                    start = 0
                    
                    self.similarities.removeAll()
                    
                    for _ in stride(from: 0, to: sims.count, by: self.textualItems.count) {
                        self.similarities.append(Array(sims[start...start+self.textualItems.count - 1]))
                        start += self.textualItems.count
                    }
                    
                    let similarities = self.createSimilarities()
                    
                    #if DEBUG
                    print("FROM CLUSTERING - ADD - ALL PAGES AFTER ADDING: ", textualItem.uuid.description, " FROM Tab ID: ", textualItem.tabId.description)
                    for val in self.textualItems {
                        print("FROM CLUSTERING - ADD - UUID: ", val.uuid)
                        print("FROM CLUSTERING - ADD - TABID: ", val.tabId)
                        print("FROM CLUSTERING - ADD - URL: ", val.url)
                        print("FROM CLUSTERING - ADD - Title: ", val.title)
                        print("FROM CLUSTERING - ADD - Processed Title: ", val.processTitle())
                        print("FROM CLUSTERING - ADD - Content: ", val.content[val.content.startIndex..<String.Index(utf16Offset:min(val.content.count, 100), in: val.content)])
                        print("--------")
                    }
                    print("FROM CLUSTERING - ADD - Similarities: ", self.similarities)
                    #endif
                    let end = DispatchTime.now()
                    let nanoTime = end.uptimeNanoseconds - startTime.uptimeNanoseconds
                    print("Time elapsed: \(Float(nanoTime / 1000000)) ms.")
                            
                    continuation.resume(returning: (pageGroups: self.pagesClusters, noteGroups: self.notesClusters, similarities: similarities))
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
    /// - Returns: - pageGroups: Newly computed pages cluster.
    ///            - noteGroups: Newly computed notes cluster.
    public func changeCandidate(expectedClusters: [[TextualItem]]) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]], similarities: [UUID: [UUID: Float]]) {
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<([[UUID]], [[UUID]], [UUID: [UUID: Float]]), Error>) in
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
                    
                    indices = Array(UnsafeBufferPointer(start: result.cluster.pointee.indices, count: Int(result.cluster.pointee.indices_size)))
                    clusters_split = Array(UnsafeBufferPointer(start: result.cluster.pointee.clusters_split, count: Int(result.cluster.pointee.clusters_split_size)))
                    let sims = Array(UnsafeBufferPointer(start: result.similarities, count: self.textualItems.count * self.textualItems.count))
                    var start = 0
                    
                    self.pagesClusters.removeAll()
                    self.notesClusters.removeAll()
                    
                    for split in clusters_split {
                        var clusterPage = [UUID]()
                        var clusterNote = [UUID]()
                        
                        for idx in start...(start + Int(split)) - 1 {
                            if self.textualItems[Int(indices[Int(idx)])].type == .page {
                                clusterPage.append(self.textualItems[Int(indices[Int(idx)])].uuid)
                            } else {
                                clusterNote.append(self.textualItems[Int(indices[Int(idx)])].uuid)
                            }
                        }

                        start += Int(split)
                        
                        self.pagesClusters.append(clusterPage)
                        self.notesClusters.append(clusterNote)
                    }
                      
                    start = 0
                    
                    self.similarities.removeAll()
                    
                    for _ in stride(from: 0, to: sims.count, by: self.textualItems.count) {
                        self.similarities.append(Array(sims[start...start+self.textualItems.count - 1]))
                        start += self.textualItems.count
                    }
                    
                    let similarities = self.createSimilarities()
        
        
                    continuation.resume(returning: (pageGroups: self.pagesClusters, noteGroups: self.notesClusters, similarities: similarities))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
