import Foundation
import NaturalLanguage

struct EntitiesInText {
    var entities = ["PersonalName": [String](), "PlaceName": [String](), "OrganizationName": [String]()]
}

public struct Page {
    public init(id: UInt64, parentId: UInt64? = nil, title: String? = nil, content: String? = nil) {
        self.id = id
        self.parentId = parentId
        self.title = title
        self.content = content
    }

    var id: UInt64
    var parentId: UInt64?
    var title: String?
    var content: String?
    var textEmbedding: [Double]?
    var entities: EntitiesInText?
    var language: NLLanguage?
    var entitiesInTitle: EntitiesInText?
    var attachedPages = [UInt64]()
}

public struct ClusteringNote {
    public init(id: UUID, title: String? = nil, content: String? = nil) {
        self.id = id
        self.title = title
        self.content = content
    }
    var id: UUID
    var title: String?
    var content: String?  // Text in the note.
                          // TODO: Should we save to source (copy-paste from a page, user input...)
    var textEmbedding: [Double]?
    var entities: EntitiesInText?
    var language: NLLanguage?
    var entitiesInTitle: EntitiesInText?
}
