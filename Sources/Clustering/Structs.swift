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

public struct InformationForId: Equatable {
    public var title: String
    public var content: String
    
    public var isEmpty: Bool {
        self.title.isEmpty && self.content.isEmpty
    }
    
    public init(title: String, content: String) {
        self.title = title
        self.content = content
    }
    
    public init() {
        self.title = ""
        self.content = ""
    }
}

public struct Page {
    public init(id: UUID, url: URL, title: String, content: String) {
        self.id = id
        self.url = url
        self.title = title
        self.content = content
        self.contentEmbedding = []
        self.titleEmbedding = []
        self.meanTitleContentEmbedding = []
        self.processTitle()
        assert(!(self.content.isEmpty && self.title.isEmpty))
    }
    
    public init(id: UUID, url: URL, title: String) {
        self.id = id
        self.url = url
        self.title = title
        self.content = ""
        self.contentEmbedding = []
        self.titleEmbedding = []
        self.meanTitleContentEmbedding = []
        self.processTitle()
        assert(!self.title.isEmpty)
    }
    
    public init(id: UUID, url: URL, content: String) {
        self.id = id
        self.url = url
        self.title = ""
        self.content = content
        self.contentEmbedding = []
        self.titleEmbedding = []
        self.meanTitleContentEmbedding = []
        assert(!self.content.isEmpty)
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
            
            self.title = (titleAsArray.map { $0.capitalized }).joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
    
    public var isOnlyTitle: Bool {
        self.content.isEmpty
    }
    
    public var isOnlyContent: Bool {
        self.title.isEmpty
    }
    
    public var isTitleAndContent: Bool {
        !self.title.isEmpty && !self.content.isEmpty
    }
    
    public var isEmptyEmbedding: Bool {
        self.titleEmbedding.isEmpty && self.contentEmbedding.isEmpty && self.meanTitleContentEmbedding.isEmpty
    }

    public var id: UUID
    public var url: URL
    var title: String
    var content: String
    var contentEmbedding: [Double]
    var titleEmbedding: [Double]
    var meanTitleContentEmbedding: [Double]
}

public struct ClusteringNote {
    public init(id: UUID, title: String, content: String) {
        self.id = id
        self.title = title
        self.content = content
        self.contentEmbedding = []
        self.titleEmbedding = []
        self.meanTitleContentEmbedding = []
        self.processTitle()
        assert(!self.content.isEmpty && !self.title.isEmpty)
    }
    
    public init(id: UUID, title: String) {
        self.id = id
        self.title = title
        self.content = ""
        self.contentEmbedding = []
        self.titleEmbedding = []
        self.meanTitleContentEmbedding = []
        self.processTitle()
        assert(!self.content.isEmpty && !self.title.isEmpty)
    }
    
    public init(id: UUID, content: String) {
        self.id = id
        self.title = ""
        self.content = content
        self.contentEmbedding = []
        self.titleEmbedding = []
        self.meanTitleContentEmbedding = []
        assert(!self.content.isEmpty && !self.title.isEmpty)
    }
    
    mutating func processTitle() {
        let regex = try! NSRegularExpression(pattern: "\\s*[-\\|:]\\s+")
        let splitTitle = regex.splitn(self.title)
        var titleAsArray: ArraySlice<String> = []
        
        if splitTitle.count > 1 {
            titleAsArray = splitTitle[0..<splitTitle.count-1]
        } else {
            titleAsArray = splitTitle[0..<1]
        }
        
        self.title = (titleAsArray.map { $0.capitalized }).joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    public var isOnlyTitle: Bool {
        self.content.isEmpty
    }
    
    public var isOnlyContent: Bool {
        self.title.isEmpty
    }
    
    public var isTitleAndContent: Bool {
        !self.title.isEmpty && !self.content.isEmpty
    }
    
    public var id: UUID
    var title: String
    var content: String
    var contentEmbedding: [Double]
    var titleEmbedding: [Double]
    var meanTitleContentEmbedding: [Double]
}
