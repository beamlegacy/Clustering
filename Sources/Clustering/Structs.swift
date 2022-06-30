import Foundation


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
    public init(uuid: UUID, url: String = "", title: String = "", content: String = "", type: TextualItemType) {
        self.uuid = uuid
        self.url = url
        self.title = title
        self.content = content.trimmingCharacters(in: .whitespacesAndNewlines)
        self.embedding = []
        self.type = type
        
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

    public let uuid: UUID
    public let url: String
    var title: String
    let content: String
    var embedding: [Double]
    let type: TextualItemType
}
