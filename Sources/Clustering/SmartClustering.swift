import Foundation
import Accelerate
import CClustering


enum CClusteringError: Error {
    case tokenizerError
    case modelError
    case clusteringError
}


class ModelInference {
    let hidden_size: UInt16 = 384
    var model: UnsafeMutableRawPointer!
    var tokenizer: UnsafeMutableRawPointer!
    
    func prepare() {
        self.prepareModel()
        self.prepareTokenizer()
    }
    
    private func prepareModel() {
        if self.model != nil {
            return
        }
        
        guard let modelPath = Bundle.module.path(forResource: "model-optimized-int32-quantized", ofType: "onnx", inDirectory: "Resources") else {
          fatalError("Resources not found")
        }
        
        let bytesModel = modelPath.utf8CString
        //let bytesTokenizer = tokenizerModelPath.utf8CString
        
        bytesModel.withUnsafeBufferPointer { ptrModel in
            self.model = createModel(ptrModel.baseAddress, self.hidden_size)
        }
        
        // The comments below represents the way to do use UTF-8 C Strings with >= Swift 5.6.1. The day we will switch
        // to this version we could uncomment this part.
        /*modelPath.withUTF8 { cModelPath in
            model = createModel(ptrModel.baseAddress, 384)
        }*/
    }
    
    private func prepareTokenizer() {
        if self.tokenizer != nil {
            return
        }
        
        guard let tokenizerModelPath = Bundle.module.path(forResource: "sentencepiece", ofType: "bpe.model", inDirectory: "Resources")
        else {
          fatalError("Resources not found")
        }
        
        let bytesTokenizer = tokenizerModelPath.utf8CString
        
        bytesTokenizer.withUnsafeBufferPointer { ptrTokenizer in
            self.tokenizer = createTokenizer(ptrTokenizer.baseAddress, 128)
        }
        
        // The comments below represents the way to do use UTF-8 C Strings with >= Swift 5.6.1. The day we will switch
        // to this version we could uncomment this part.
        /*tokenizerModelPath.withUTF8 { cTokenizerModelPath in
            tokenizer = createTokenizer(ptrTokenizer.baseAddress, 128)
        }
        }*/
    }
    
    @MainActor func encode(tokenizerResult: inout TokenizerResult) async throws -> [Float] {
        var result = ModelResult()
        var ret: Int32 = -1
        
        ret = predict(self.model, &tokenizerResult, &result)
            
        if ret == 0 {
            return Array(UnsafeBufferPointer(start: result.weigths, count: Int(result.size)))
        }
        
        throw CClusteringError.modelError
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
    var textualItems = [TextualItem]()
    var pagesClusters = [[UUID]]()
    var notesClusters = [[UUID]]()
    var similarities = [[Float]]()
    let lock = NSLock()
    var clustering: UnsafeMutableRawPointer!
    @MainActor let modelInf = ModelInference()
    let websitesToUseOnlyTitle = ["youtube"]

    public init() {
        self.clustering = createClustering()
    }
    
    public func prepare() {
        self.modelInf.prepare()
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
        
        self.textualItems.remove(at: textualItemIndex)
        
        if type == TextualItemType.page {
            self.pagesClusters[coordinates.clusterIndex].remove(at: coordinates.indexInCluster)
        } else {
            self.notesClusters[coordinates.clusterIndex].remove(at: coordinates.indexInCluster)
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
        self.lock.lock()
        
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
        
        self.lock.unlock()
        
        return (pageGroups: self.pagesClusters, noteGroups: self.notesClusters, similarities: similarities)
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
        self.lock.lock()
        
        repeat {
        } while self.modelInf.tokenizer == nil || self.modelInf.model == nil
        let start_time = DispatchTime.now()
        #if DEBUG
        print("FROM CLUSTERING - ADD - ADDING PAGE: ", textualItem.uuid.description, " FROM Tab ID: ", textualItem.tabId.description)
        #endif
        
        let idx = self.findTextualItemIndex(of: textualItem.uuid, from: textualItem.tabId)
        
        if idx != -1 {
            #if DEBUG
            print("FROM CLUSTERING - ADD - UUID: ", textualItem.uuid.description, " FROM Tab ID: ", textualItem.tabId.description, " already exists - delete first")
            #endif
            
            _ = try self.removeActualTextualItem(textualItemIndex: idx, textualItemTabId: textualItem.tabId)
            
            self.textualItems.insert(textualItem, at: idx)
        } else {
            self.textualItems.append(textualItem)
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
        
        if text == "</s></s>" {
            self.textualItems[self.textualItems.count - 1].updateEmbedding(newEmbedding: [Float](repeating: 0.0, count: Int(self.modelInf.hidden_size)))
        } else {
            var tokenizedText = try await self.modelInf.tokenizeText(text: text)
            let embedding = try await self.modelInf.encode(tokenizerResult: &tokenizedText)
            
            self.textualItems[self.textualItems.count - 1].updateEmbedding(newEmbedding: embedding)
        }
        
        var result = ClusteringResult()
        var ebds = [UnsafePointer<Float>?]()
        var ret: Int32 = -1
        var i = 0
        
        while i < self.textualItems.count {
            var t = self.textualItems[i].embedding
            ebds.append(&t)
            i += 1
        }
        
        ret = create_clusters(self.clustering, &ebds, self.modelInf.hidden_size, UInt16(self.textualItems.count), &result)
        
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
        let nanoTime = end.uptimeNanoseconds - start_time.uptimeNanoseconds
        print("Time elapsed: \(Float(nanoTime / 1000000)) ms.")
        self.lock.unlock()
                
        return (pageGroups: self.pagesClusters, noteGroups: self.notesClusters, similarities: similarities)
            
    }

    /// Find a better threshold and recompute the clusters.
    ///
    /// - Parameters:
    ///      - expectedClusters: The gold representation of the clusters as expected
    /// - Returns: - pageGroups: Newly computed pages cluster.
    ///            - noteGroups: Newly computed notes cluster.
    public func changeCandidate(expectedClusters: inout ClusterDefinition) async throws -> (pageGroups: [[UUID]], noteGroups: [[UUID]], similarities: [UUID: [UUID: Float]]) {
        self.lock.lock()
        var result = ClusteringResult()
        var ret: Int32 = -1
        
        ret = recompute_clustering_threshold(self.clustering, &expectedClusters, &result)
        
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
        
        self.lock.unlock()
        
        return (pageGroups: self.pagesClusters, noteGroups: self.notesClusters, similarities: similarities)
    }
}
