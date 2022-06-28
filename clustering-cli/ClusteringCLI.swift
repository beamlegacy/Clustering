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
    
    @Flag(help: "Debug mode.")
    var debug = false
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
        var pages: [UUID: Page] = [:]
        let csvFile = try CSVReader.decode(input: URL(fileURLWithPath: inputFile)){ $0.headerStrategy = .firstLine }
        var clusteredPageIds: [[UUID]] = []
        var id2colours: [String: String] = [:]
        
        for row in csvFile.records {
            if row["noteName"] == "<???>" {
                if let isOpenAtExport = row["isOpenAtExport"], let pageId = row["id"], let title = row["title"], let originalContent = row["cleanedContent"], let url = row["url"], let tabColouringGroupId = row["tabColouringGroupId"], let userCorrectionGroupId = row["userCorrectionGroupId"] {
                    
                    guard let convertedPageId = UUID(uuidString: pageId) else {
                        return
                    }
                    let convertedIsOpenAtExport = isOpenAtExport == "false" ? false: true
                    let cleanedTitle = title.replacingOccurrences(of: " and some text", with: "").trimmingCharacters(in: .whitespaces)
                    let cleanedURL = url == "<???>" ? "http://empty": url
                    
                    if convertedIsOpenAtExport && !(tabColouringGroupId == "<???>" && userCorrectionGroupId == "<???>") {
                        if tabColouringGroupId != "<???>" && userCorrectionGroupId == "<???>" {
                            id2colours[pageId] = tabColouringGroupId
                        } else if (tabColouringGroupId == "<???>" && userCorrectionGroupId != "<???>") ||
                                  (tabColouringGroupId != "<???>" && userCorrectionGroupId != "<???>") {
                            id2colours[pageId] = userCorrectionGroupId
                        }
                        
                        guard let unwrappedURL = URL(string: cleanedURL) else {
                            return
                        }
                        pages[convertedPageId] = Page(id: convertedPageId, url: unwrappedURL, title: cleanedTitle, content: originalContent)
                    }
                }
            }
        }
        
        for page in pages.values {
            if self.debug {
                print("Add Page: " + page.id.description)
                fflush(stdout)
            }
            
            clusteredPageIds = try await cluster.add(page: page).pageGroups
        }
        
        if self.debug {
            for c in clusteredPageIds.enumerated() {
                print("Cluster \(c.offset):")
                for p in c.element {
                    if let u = pages[p] {
                        print("\t\(u.url)")
                        fflush(stdout)
                    }
                }
            }
        }
        
        var outputCsv: [[String]] = [csvFile.headers]
        var groupId2colours: [String:String] = [:]
        
        if let columns = csvFile[column: "id"] {
            for rowId in columns.enumerated() {
                if csvFile[rowId.offset][11] == "false" || (csvFile[rowId.offset][4] == "<???>" && csvFile[rowId.offset][5] == "<???>")  {
                    outputCsv.append(csvFile[rowId.offset])
                } else {
                    if let convertedId = UUID(uuidString: rowId.element) {
                        var newRow: [String] = []
                        
                        newRow.append(csvFile[rowId.offset][0])
                        newRow.append(csvFile[rowId.offset][1])
                        newRow.append(csvFile[rowId.offset][2])
                        newRow.append(csvFile[rowId.offset][3])
                        
                        let groupId = self.findGroupForID(id: convertedId, groups: clusteredPageIds)
                        var currentColor = ""
                        
                        if groupId != -1 {
                            if let color = groupId2colours[String(groupId)] {
                                currentColor = color
                                newRow.append(color)
                            } else if let color = id2colours[rowId.element] {
                                currentColor = color
                                newRow.append(color)
                                groupId2colours[String(groupId)] = color
                                
                                for (key, value) in id2colours where value == color {
                                    id2colours.removeValue(forKey: key)
                                }
                            } else {
                                currentColor = UUID().description
                                groupId2colours[String(groupId)] = currentColor
                                newRow.append(currentColor)
                            }
                        } else {
                            currentColor = UUID().description
                            groupId2colours[String(groupId)] = currentColor
                            newRow.append(currentColor)
                        }
                        
                        if csvFile[rowId.offset][4] == currentColor || csvFile[rowId.offset][5] == currentColor {
                            newRow.append("<???>")
                        } else if csvFile[rowId.offset][5] == "<???>" && csvFile[rowId.offset][4] != currentColor {
                            newRow.append(csvFile[rowId.offset][4])
                        } else if (csvFile[rowId.offset][5] != currentColor && csvFile[rowId.offset][4] != currentColor) {
                            newRow.append(csvFile[rowId.offset][5])
                        }
                        
                        newRow.append(csvFile[rowId.offset][6])
                        newRow.append(csvFile[rowId.offset][7])
                        newRow.append(csvFile[rowId.offset][8])
                        newRow.append(csvFile[rowId.offset][9])
                        newRow.append(csvFile[rowId.offset][10])
                        newRow.append(csvFile[rowId.offset][11])
                        newRow.append(csvFile[rowId.offset][12])
                        newRow.append(csvFile[rowId.offset][13])
                        
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
