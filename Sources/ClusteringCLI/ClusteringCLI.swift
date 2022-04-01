//
//  ClusteringCLI.swift
//
//
//  Created by Julien Plu on 22/03/2022.
//

import ArgumentParser
import Foundation
import Clustering
import SwiftCSV
import CodableCSV

@main
struct ClusteringCLI: ParsableCommand {
    @Option(help: "The CSV file to inject in the Clustering package.")
    var inputFile: String
    
    @Option(help: "The CSV output file where the replayed export will be saved.")
    var outputFile: String
}

extension ClusteringCLI {
    func findGroupForID(id: UUID, groups: [[UUID]]) -> Int {
        for group in groups.enumerated() {
            if group.element.contains(id) {
                return group.offset
            }
        }
        
        return -1
    }
    
    func run() throws {
        let cluster = Cluster(useMainQueue: false)
        var pages: [Page] = []
        var notes: [ClusteringNote] = []
        let csvFile = try CSVReader.decode(input: URL(fileURLWithPath: inputFile)){ $0.headerStrategy = .firstLine }
        var clusteredPageIds: [[UUID]] = []
        var clusteredNoteIds: [[UUID]] = []
        let runLoop = CFRunLoopGetCurrent()
        var id2colours: [String: String] = [:]
        
        for row in csvFile.records {
            if let id = row["Id"], let tabColouringGroupId = row["tabColouringGroupId"], let userCorrectionGroupId = row["userCorrectionGroupId"] {
                if tabColouringGroupId != "<???>" {
                    if userCorrectionGroupId == "<???>" {
                        id2colours[id] = tabColouringGroupId
                    }
                }
            }
            if row["noteName"] != "<???>" {
                if let noteId = row["Id"], let title = row["title"], let content = row["cleanedContent"] {
                    guard let convertedPageId = UUID(uuidString: noteId) else {
                        return
                    }
                    notes.append(ClusteringNote(id: convertedPageId, title: title, content: [content]))
                }
            } else {
                if let pageId = row["Id"], let parentId = row["parentId"], let title = row["title"], let cleanedContent = row["cleanedContent"], let url = row["url"] {
                    guard let convertedPageId = UUID(uuidString: pageId) else {
                        return
                    }
                    let convertedParentId = parentId == "<???>" ? nil: UUID(uuidString: parentId)
                    
                    pages.append(Page(id: convertedPageId, parentId: convertedParentId, url: URL(string: url), title: title, cleanedContent: cleanedContent))
                }
            }
        }
        
        DispatchQueue.global().async {
            let semaphore = DispatchSemaphore(value: 0)
            
            for note in notes {
                cluster.add(note: note, ranking: nil, completion: { result in
                    switch result {
                    case .failure:
                        ()
                    case .success(let result):
                        clusteredNoteIds = result.noteGroups
                    }
                    semaphore.signal()
                })
                semaphore.wait()
            }
            
            for page in pages {
                cluster.add(page: page, ranking: nil, completion: { result in
                    switch result {
                    case .failure:
                        ()
                    case .success(let result):
                        clusteredPageIds = result.pageGroups
                    }
                    semaphore.signal()
                })
                semaphore.wait()
            }
            
            CFRunLoopStop(runLoop)
        }
        
        CFRunLoopRun()
        
        for note in notes.enumerated() {
            if findGroupForID(id: note.element.id, groups: clusteredNoteIds) == -1 {
                clusteredNoteIds.insert([], at: note.offset)
            }
        }
        
        for page in pages.enumerated() {
            if findGroupForID(id: page.element.id, groups: clusteredPageIds) == -1 {
                clusteredPageIds.insert([], at: page.offset)
            }
        }
        
        var outputCsv: [[String]] = [csvFile.headers]
        let allGroups = clusteredNoteIds + clusteredPageIds
        var groupId2colours: [String:String] = [:]
        
        if let columns = csvFile[column: "Id"] {
            for rowId in columns.enumerated() {
                if let convertedId = UUID(uuidString: rowId.element) {
                    let info = cluster.getExportInformationForId(id: convertedId)
                    
                    if info.title == nil {
                        outputCsv.append(csvFile[rowId.offset])
                    } else {
                        var newRow: [String] = []
                        
                        newRow.append(csvFile[rowId.offset][0])
                        newRow.append(csvFile[rowId.offset][1])
                        
                        let groupId = findGroupForID(id: convertedId, groups: allGroups)
                        
                        newRow.append(String(groupId))
                        newRow.append(String(findGroupForID(id: convertedId, groups: clusteredPageIds)))
                        
                        if csvFile[rowId.offset][4] == "<???>" && csvFile[rowId.offset][5] == "<???>" {
                            newRow.append(csvFile[rowId.offset][4])
                            newRow.append(csvFile[rowId.offset][5])
                        } else {
                            var currentColor = ""
                            if let color = groupId2colours[String(groupId)] {
                                currentColor = color
                                newRow.append(color)
                                groupId2colours[String(groupId)] = color
                            } else if let color = id2colours[rowId.element] {
                                currentColor = color
                                newRow.append(color)
                                groupId2colours[String(groupId)] = color
                                for (key, value) in id2colours where value == color {
                                    id2colours.removeValue(forKey: key)
                                }
                            } else {
                                currentColor = UUID().description
                                newRow.append(currentColor)
                            }
                            
                            if csvFile[rowId.offset][4] == currentColor && csvFile[rowId.offset][5] == "<???>" {
                                newRow.append("<???>")
                            } else if csvFile[rowId.offset][5] == currentColor {
                                newRow.append("<???>")
                            } else if csvFile[rowId.offset][4] != currentColor && csvFile[rowId.offset][5] == "<???>" {
                                newRow.append(csvFile[rowId.offset][4])
                            } else {
                                newRow.append(csvFile[rowId.offset][5])
                            }
                        }
                        
                        if let title = info.title, let cleanedContent = info.cleanedContent, let entities = info.entitiesInText, let entitiesInTitle = info.entitiesInTitle, let language = info.language, let parentId = info.parentId {
                            newRow.append(title)
                            newRow.append(cleanedContent)
                            newRow.append(entities.description)
                            newRow.append(entitiesInTitle.description)
                            newRow.append(language.rawValue)
                            newRow.append(csvFile[rowId.offset][11])
                            newRow.append(convertedId.description)
                            newRow.append(parentId.description)
                        } else {
                            newRow.append(csvFile[rowId.offset][6])
                            newRow.append(csvFile[rowId.offset][7])
                            newRow.append(csvFile[rowId.offset][8])
                            newRow.append(csvFile[rowId.offset][9])
                            newRow.append(csvFile[rowId.offset][10])
                            newRow.append(csvFile[rowId.offset][11])
                            newRow.append(csvFile[rowId.offset][12])
                            newRow.append(csvFile[rowId.offset][13])
                        }
                        assert(newRow.count == csvFile[rowId.offset].count)
                        outputCsv.append(newRow)
                    }
                }
            }
        }
        
        assert(outputCsv.count == csvFile.count + 1)

        try CSVWriter.encode(rows: outputCsv, into: URL(fileURLWithPath: outputFile), append: false)
    }
}
