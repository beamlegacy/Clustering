import Foundation
import NaturalLanguage

public struct EntitiesInText : Equatable {
    var entities = ["PersonalName": [String](), "PlaceName": [String](), "OrganizationName": [String]()]
    public var description: String {
        return "PER[" + entities["PersonalName"]!.description + "] - LOC[" + entities["PlaceName"]!.description + "] - ORG[" + entities["OrganizationName"]!.description + "]"
    }
}

public struct InformationForId: Equatable {
    public var title: String? = nil
    public var cleanedContent: String? = nil
    public var entitiesInText: EntitiesInText? = nil
    public var entitiesInTitle: EntitiesInText? = nil
    public var language: NLLanguage? = nil
    public var parentId: UUID? = nil
}

public struct Page {
    public init(id: UUID, parentId: UUID? = nil, url: URL? = nil, title: String? = nil, originalContent: [String]? = nil, cleanedContent: String? = nil) {
        self.id = id
        self.parentId = parentId
        self.title = title
        self.originalContent = originalContent
        self.cleanedContent = cleanedContent
        self.url = url
    }

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
}

public struct ClusteringNote {
    public init(id: UUID, title: String? = nil, content: [String]? = nil) {
        self.id = id
        self.title = title
        self.originalContent = content
    }
    
    public var id: UUID
    var title: String?
    var originalContent: [String]?  // Text in the note.
                          // TODO: Should we save to source (copy-paste from a page, user input...)
    var cleanedContent: String?
    var textEmbedding: [Double]?
    var entities: EntitiesInText?
    var language: NLLanguage?
    var entitiesInTitle: EntitiesInText?
}
