//
//  main.swift
//
//
//  Created by Julien Plu on 22/03/2022.
//

import ArgumentParser
import Foundation
import Clustering
import SwiftCSV


@main
struct ClusteringCLI: ParsableCommand {
    @Argument(help: "The CSV file to inject in the Clustering package.")
    var inputFile: String
}

extension ClusteringCLI {
    mutating func run() throws {
        let cluster = Cluster()
        var pages: [Page] = []
        var notes: [ClusteringNote] = []
        var noteIds: [UUID] = []
        var pageIds: [UUID] = []
        let csvFile: CSV = try CSV(url: URL(fileURLWithPath: inputFile))
        
        try csvFile.enumerateAsDict { dict in
            if dict["pageId"] == "<???>" {
                if let title = dict["title"], let content = dict["cleanedContent"] {
                    let currentId = UUID()
                    noteIds.append(currentId)
                    notes.append(ClusteringNote(id: currentId, title: title, content: [content]))
                }
            } else {
                if let pageId = dict["pageId"], let parentId = dict["parentId"], let title = dict["title"], let cleanedContent = dict["cleanedContent"], let url = dict["url"] {
                    guard let convertedPageId = UUID(uuidString: pageId) else {
                        return
                    }
                    pageIds.append(convertedPageId)
                    let convertedParentId = parentId == "<???>" ? nil: UUID(uuidString: parentId)
                    
                    pages.append(Page(id: convertedPageId, parentId: convertedParentId, url: URL(string: url), title: title, cleanedContent: cleanedContent))
                }
            }
        }
        
        //var semaphore = DispatchSemaphore(value: 0)
        var clusteringNotes = 0
        let myGroup = DispatchGroup()
        for note in notes.enumerated() {
            myGroup.enter()
            cluster.add(note: note.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    print(error)
                    clusteringNotes +=  1
                    myGroup.leave()
                case .success(let result):
                    _ = result.0
                    clusteringNotes +=  1
                    myGroup.leave()
                }
                
                //if clusteringNotes == notes.count {
                //    semaphore.signal()
                //}
            })
        }

        //semaphore.wait()
        myGroup.notify(queue: .main) {
            print("Finished all requests.")
        }

        let semaphore = DispatchSemaphore(value: 0)
        var clusteringPages = 0
        
        for page in pages.enumerated() {
            cluster.add(page: page.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    print(error)
                    clusteringPages += 1
                case .success(let result):
                    _ = result.0
                    clusteringPages += 1
                }
                if clusteringPages == pages.count {
                    semaphore.signal()
                }
            })
        }
        
        semaphore.wait()
        
        for nid in noteIds {
            print(cluster.getExportInformationForId(id: nid))
        }
        
        for pid in pageIds {
            print(cluster.getExportInformationForId(id: pid))
        }
    }
}
