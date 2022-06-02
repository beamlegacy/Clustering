//
//  ClusteringCLI.swift
//
//
//  Created by Julien Plu on 22/03/2022.
//

// Issue with the @main annotation https://github.com/apple/swift/issues/55127

import ArgumentParser
import Foundation
import Clustering
import CodableCSV
import CClustering


struct StandardErrorOutputStream: TextOutputStream {
    let stderr = FileHandle.standardError

    func write(_ string: String) {
        guard let data = string.data(using: .utf8) else {
            fatalError() // encoding failure: handle as you wish
        }
        stderr.write(data)
    }
}

@main
struct ClusteringCLI: AsyncParsableCommand {
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
    
    func allContentsOfDirectory(atPath path: String) -> [String] {
        var paths = [String]()
        do {
            let url = URL(fileURLWithPath: path)
            let contents = try FileManager.default.contentsOfDirectory(atPath: path)
            
            for content in contents {
                let contentUrl = url.appendingPathComponent(content)
                if contentUrl.hasDirectoryPath {
                    paths.append(contentsOf: allContentsOfDirectory(atPath: contentUrl.path))
                }
                else {
                    paths.append(contentUrl.path)
                }
            }
        }
        catch {}
        return paths
    }
    
    mutating func run() async throws {
        let cluster = Cluster()
        var pages: [Page] = []
        var notes: [ClusteringNote] = []
        let csvFile = try CSVReader.decode(input: URL(fileURLWithPath: inputFile)){ $0.headerStrategy = .firstLine }
        var clusteredPageIds: [[UUID]] = []
        var clusteredNoteIds: [[UUID]] = []
        var id2colours: [String: String] = [:]
        
        for row in csvFile.records {
            if let id = row["id"], let tabColouringGroupId = row["tabColouringGroupId"], let userCorrectionGroupId = row["userCorrectionGroupId"] {
                if tabColouringGroupId != "<???>" {
                    if userCorrectionGroupId == "<???>" {
                        id2colours[id] = tabColouringGroupId
                    }
                }
            }
            if row["noteName"] != "<???>" {
                if let noteId = row["id"], let title = row["title"], let content = row["cleanedContent"] {
                    guard let convertedPageId = UUID(uuidString: noteId) else {
                        return
                    }
                    notes.append(ClusteringNote(id: convertedPageId, title: title.replacingOccurrences(of: " and some text", with: ""), content: [content]))
                }
            } else {
                if let pageId = row["id"], let parentId = row["parentId"], let title = row["title"], let originalContent = row["cleanedContent"], let url = row["url"] {
                    guard let convertedPageId = UUID(uuidString: pageId) else {
                        return
                    }
                    let convertedParentId = parentId == "<???>" ? nil: UUID(uuidString: parentId)
                    
                    
                    pages.append(Page(id: convertedPageId, parentId: convertedParentId, url: URL(string: url), title: title.replacingOccurrences(of: " and some text", with: ""), originalContent: [originalContent]))
                }
            }
        }
        
        var result: (pageGroups: [[UUID]], noteGroups: [[UUID]], similarities: [UUID: [UUID: Double]]) = (pageGroups: [], noteGroups: [], similarities: [:])
        
        for note in notes {
            print("Add Note: " + note.id.description)
            fflush(stdout)
            result = try await cluster.add(note: note)
        }
            
        for page in pages {
            print("Add Page: " + page.id.description)
            fflush(stdout)
            result = try await cluster.add(page: page)
        }
        
        clusteredNoteIds = result.noteGroups
        clusteredPageIds = result.pageGroups
                
        var outputCsv: [[String]] = [csvFile.headers]
        var groupId2colours: [String:String] = [:]
        var emptyClusters = 0
        
        if let columns = csvFile[column: "id"] {
            for rowId in columns.enumerated() {
                if let convertedId = UUID(uuidString: rowId.element) {
                    let info = cluster.getExportInformationForId(id: convertedId)
                    
                    if info.isEmpty {
                        outputCsv.append(csvFile[rowId.offset])
                        emptyClusters += 1
                    } else {
                        var newRow: [String] = []
                        
                        newRow.append(csvFile[rowId.offset][0])
                        newRow.append(csvFile[rowId.offset][1])
                        
                        var navigationGroupId = findGroupForID(id: convertedId, groups: clusteredPageIds)
                        var groupId = -1
                        
                        if navigationGroupId != -1 {
                            groupId = navigationGroupId + emptyClusters
                        } else {
                            groupId = findGroupForID(id: convertedId, groups: clusteredNoteIds)
                            navigationGroupId = -1
                        }
                        

                        newRow.append(String(groupId))
                        newRow.append(String(navigationGroupId))
                        
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

                        newRow.append(info.title ?? csvFile[rowId.offset][6])
                        newRow.append(info.cleanedContent ?? csvFile[rowId.offset][7])
                        
                        if let entitiesInText = info.entitiesInText {
                            newRow.append(entitiesInText.description)
                        } else {
                            newRow.append(csvFile[rowId.offset][8])
                        }
                        
                        if let entitiesInTitle = info.entitiesInTitle {
                            newRow.append(entitiesInTitle.description)
                        } else {
                            newRow.append(csvFile[rowId.offset][9])
                        }
                        
                        newRow.append(csvFile[rowId.offset][10])
                        newRow.append(csvFile[rowId.offset][11])
                        newRow.append(convertedId.description)
                        
                        if let parentId = info.parentId {
                            newRow.append(parentId.description)
                        } else {
                            newRow.append(csvFile[rowId.offset][13])
                        }
                        
                        if newRow.count != csvFile[rowId.offset].count {
                            print("Err: \(newRow.count) is different from \(csvFile[rowId.offset].count)")
                            fflush(stdout)
                            return
                        }
                        
                        outputCsv.append(newRow)
                    }
                }
            }
        }
        
        if outputCsv.count == csvFile.count + 1 {
            do {
                try CSVWriter.encode(rows: outputCsv, into: URL(fileURLWithPath: outputFile), append: false)
            } catch let error {
                var errStream = StandardErrorOutputStream()
                print("\(error)", to: &errStream)
            }
        } else {
            print("Err: \(outputCsv.count) is different from \(csvFile.count)")
            fflush(stdout)
        }
    }
}
