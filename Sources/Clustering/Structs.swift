import Foundation
import NaturalLanguage

public struct EntitiesInText : Equatable {
    var entities = ["PersonalName": [String](), "PlaceName": [String](), "OrganizationName": [String]()]
    public var description: String {
        return "PER[" + entities["PersonalName"]!.description + "] - LOC[" + entities["PlaceName"]!.description + "] - ORG[" + entities["OrganizationName"]!.description + "]"
    }
    var isEmpty: Bool {
        if (entities["PersonalName"]?.count ?? 0) + (entities["PlaceName"]?.count ?? 0) + (entities["OrganizationName"]?.count ?? 0) == 0 {
            return true
        } else {
            return false
        }
    }
}
func + (left: EntitiesInText, right: EntitiesInText) -> EntitiesInText {
    var totalEntities = EntitiesInText()
    totalEntities.entities["PersonalName"] = (left.entities["PersonalName"] ?? []) + (right.entities["PersonalName"] ?? [])
    totalEntities.entities["PlaceName"] = (left.entities["PlaceName"] ?? []) + (right.entities["PlaceName"] ?? [])
    totalEntities.entities["OrganizationName"] = (left.entities["OrganizationName"] ?? []) + (right.entities["OrganizationName"] ?? [])
    return totalEntities
}

public struct InformationForId: Equatable {
    public var title: String? = nil
    public var cleanedContent: String? = nil
    public var entitiesInText: EntitiesInText? = nil
    public var entitiesInTitle: EntitiesInText? = nil
    public var language: NLLanguage? = nil
    public var parentId: UUID? = nil
    
    public var isEmpty: Bool {
        title == nil && cleanedContent == nil
    }

}

public struct Page {
    public var id: UUID
    var parentId: UUID?
    var title: String?
    var originalContent: [String]?
    var cleanedContent: String?
    var textEmbedding: [Double]?
    var entities: EntitiesInText?
    var language: NLLanguage?
    var entitiesInTitle: EntitiesInText?
    var url: URL?
    var domain: String?
    var beWith: [UUID]?
    var beApart: [UUID]?
    
    public init(id: UUID, parentId: UUID? = nil, url: URL? = nil, title: String? = nil, originalContent: [String]? = nil, cleanedContent: String? = nil, language: NLLanguage? = nil, beWith: [UUID]? = nil, beApart: [UUID]? = nil) {
        self.id = id
        self.parentId = parentId
        self.title = title
        self.originalContent = originalContent
        self.cleanedContent = cleanedContent
        self.url = url
        self.language = language
        self.beWith = beWith
        self.beApart = beApart
    }
    
    public func toTextualItem() -> TextualItem {
        let urlStr: String = self.url?.description ?? ""
        return TextualItem(id: self.id, url: urlStr, title: self.title ?? "", originalContent: self.originalContent, cleanedContent: self.cleanedContent, type: TextualItemType.page, parentId: self.parentId, language: self.language, beWith: self.beWith, beApart: self.beApart)
    }
}

public struct ClusteringNote {
    public var id: UUID
    var title: String?
    var originalContent: [String]?  // Text in the note.
                          // TODO: Should we save to source (copy-paste from a page, user input...)
    var cleanedContent: String?
    var textEmbedding: [Double]?
    var entities: EntitiesInText?
    var language: NLLanguage?
    var entitiesInTitle: EntitiesInText?
    
    public init(id: UUID, title: String? = nil, content: [String]? = nil, language: NLLanguage? = nil) {
        self.id = id
        self.title = title
        self.originalContent = content
        self.language = language
    }
    
    public func toTextualItem() -> TextualItem {
        return TextualItem(id: self.id, title: self.title ?? "", originalContent: self.originalContent, type: TextualItemType.note, language: self.language)
    }
}



extension NSRegularExpression {
    /// An array of substring of the given string, separated by this regular expression, restricted to returning at most n items.
    /// If n substrings are returned, the last substring (the nth substring) will contain the remainder of the string.
    /// - Parameter str: String to be matched
    /// - Parameter n: If `n` is specified and n != -1, it will be split into n elements else split into all occurences of this pattern
    func splitn(_ str: String, _ n: Int = -1) -> [String] {
        let range = NSRange(location: 0, length: str.count)
        let matches = self.matches(in: str, range: range);
        var result = [String]()
        
        if (n != -1 && n < 2) || matches.isEmpty  {
            return [str]
        }
        
        if let first = matches.first?.range {
            if first.location == 0 {
                result.append("")
            }
            
            if first.location != 0 {
                let _range = NSRange(location: 0, length: first.location)
            
                result.append(String(str[Range(_range, in: str)!]))
            }
        }
        
        for (cur, next) in zip(matches, matches[1...]) {
            let loc = cur.range.location + cur.range.length
            
            if n != -1 && result.count + 1 == n {
                let _range = NSRange(location: loc, length: str.count - loc)
            
                result.append(String(str[Range(_range, in: str)!]))
                
                return result
                
            }
            
            let len = next.range.location - loc
            let _range = NSRange(location: loc, length: len)
            
            result.append(String(str[Range(_range, in: str)!]))
        }
        
        if let last = matches.last?.range, !(n != -1 && result.count >= n) {
            let lastIndex = last.length + last.location
            
            if lastIndex == str.count {
                result.append("")
            }
            
            if lastIndex < str.utf8.count {
                let _range = NSRange(location: lastIndex, length: str.count - lastIndex)
            
                result.append(String(str[Range(_range, in: str)!]))
            }
        }
        
        return result;
    }
}

public enum TextualItemType {
    case page
    case note
}


public struct TextualItem: Equatable {
    public let uuid: UUID
    public let url: String
    let parentId: UUID?
    let language: NLLanguage?
    let beApart: [UUID]?
    let beWith: [UUID]?
    var title: String
    let content: String
    var embedding: [Double]
    let type: TextualItemType
    let cleanedContent: String?
    let originalContent: [String]?
    
    public init(id: UUID, url: String = "", title: String = "", originalContent: [String]? = nil, cleanedContent: String? = nil, type: TextualItemType, parentId: UUID? = nil, language: NLLanguage? = nil, beWith: [UUID]? = nil, beApart: [UUID]? = nil) {
        self.uuid = id
        self.url = url
        self.title = title
        self.originalContent = originalContent
        self.cleanedContent = cleanedContent
        
        if let originalContentUnwraped = self.originalContent {
            self.content = originalContentUnwraped.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        } else if let cleanedContentUnwraped = cleanedContent {
            self.content = cleanedContentUnwraped
        } else {
            self.content = ""
        }
        
        self.embedding = []
        self.type = type
        self.language = language
        self.parentId = parentId
        self.beWith = beWith
        self.beApart = beApart
        
        self.processTitle()
    }
    
    mutating func processTitle() {
        if !self.title.isEmpty {
            let regex = try! NSRegularExpression(pattern: "\\s*[-\\|:\\(]\\s+")
            let splitTitle = regex.splitn(self.title)
            var titleAsArray: ArraySlice<String> = []
            
            if splitTitle.count > 1 {
                titleAsArray = splitTitle[0..<splitTitle.count-1]
            } else {
                titleAsArray = splitTitle[0..<1]
            }
            
            self.title = titleAsArray.joined(separator: " ").capitalized.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    mutating func updateEmbedding(newEmbedding: [Double]) {
        self.embedding = newEmbedding
    }
    
    func toPage() -> Page {
        return Page(id: self.uuid, parentId: self.parentId, url: URL(string: self.url), title: self.title, originalContent: self.originalContent, cleanedContent: cleanedContent, language: self.language, beWith: self.beWith, beApart: self.beApart)
    }
    
    func toNote() -> ClusteringNote {
        return ClusteringNote(id: self.uuid, title: self.title, content: self.originalContent, language: self.language)
    }
}
