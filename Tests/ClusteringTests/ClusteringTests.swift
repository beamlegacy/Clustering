import Nimble
import XCTest
import LASwift
import NaturalLanguage
@testable import Clustering
import Accelerate
import Clustering

// swiftlint:disable:next type_body_length
class ClusteringTests: XCTestCase {
    /// Test adding and removing of data points from a (non-navigation) similarity matrix. For both addition and removal, test that all locations in the matrix (first, last, middle) work as expected
    func testAddandRemoveDataPointsToSimilarityMatrix() throws {
        let cluster = Cluster()
        
        // Checking the state of the matrix after initialization
        expect(cluster.textualSimilarityMatrix.matrix) == Matrix([0])
        
        try cluster.textualSimilarityMatrix.addDataPoint(similarities: [0], type: .page, numExistingNotes: 0, numExistingPages: 0)
        
        // Checking the first page doesn't influence the matrix
        expect(cluster.textualSimilarityMatrix.matrix) == Matrix([0])
        
        try cluster.textualSimilarityMatrix.addDataPoint(similarities: [0.5], type: .page, numExistingNotes: 0, numExistingPages: 1)
        
        // Checking the addition of the second page, first addition to the matrix
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0.5, 0.5, 0]
        
        try cluster.textualSimilarityMatrix.addDataPoint(similarities: [0.3, 0.2], type: .page, numExistingNotes: 0, numExistingPages: 2)
        
        // Checking the addition of a third page, in position whereToAdd = .last
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0.5, 0.3, 0.5, 0, 0.2, 0.3, 0.2, 0]
        
        try cluster.textualSimilarityMatrix.addDataPoint(similarities: [0.1, 0.1, 0.1], type: .note, numExistingNotes: 0, numExistingPages: 3)
        
        // Checking the addition of a first note, in position whereToAdd = .first
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0.1, 0.1, 0.1, 0.1, 0, 0.5, 0.3, 0.1, 0.5, 0, 0.2, 0.1, 0.3, 0.2, 0]
        
        try cluster.textualSimilarityMatrix.addDataPoint(similarities: [0, 0.9, 0.9, 0.9], type: .note, numExistingNotes: 1, numExistingPages: 3)
        
        // Checking the addition of a second note, in position whereToAdd = .middle
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0, 0.1, 0.1, 0.1, 0, 0, 0.9, 0.9, 0.9, 0.1, 0.9, 0, 0.5, 0.3, 0.1, 0.9, 0.5, 0, 0.2, 0.1, 0.9, 0.3, 0.2, 0]
        
        try cluster.textualSimilarityMatrix.removeDataPoint(index: 3)
        
        // Testing removal from the middle of the matrix
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0, 0.1, 0.1, 0, 0, 0.9, 0.9, 0.1, 0.9, 0, 0.3, 0.1, 0.9, 0.3, 0]
        
        try cluster.textualSimilarityMatrix.removeDataPoint(index: 0)
        
        // Testing removal of index 0 (first position in the matrix)
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0.9, 0.9, 0.9, 0, 0.3, 0.9, 0.3, 0]
        
        try cluster.textualSimilarityMatrix.removeDataPoint(index: 2)
        
        // Testing removal of the last position in the matrix
        expect(cluster.textualSimilarityMatrix.matrix.flat) == [0, 0.9, 0.9, 0]
    }

    /// Test the spectral clustering function on a small adjacency matrix
    func testSpectralClustering() throws {
        let cluster = Cluster()
        var i = 0
        var clustersResult = [Int]()
        
        cluster.adjacencyMatrix = Matrix([[0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        repeat {
            let predictedClusters = try cluster.spectralClustering()
            clustersResult = cluster.stabilize(predictedClusters)
            i += 1
        } while clustersResult != [0, 0, 1, 1, 2, 3, 3, 4, 0, 0] && i < 1
        
        // For now it seems that resuls are very stable, if that changes the limit in the loop can be raised up to 10
        expect(clustersResult) == [0, 0, 1, 1, 2, 3, 3, 4, 0, 0]
    }

    /// Test the cosine similarity function
    func testCosineSimilarity() throws {
        let cluster = Cluster()
        let vec1 = [0.0, 1.5, 3.0, 4.5, 6.0]
        let vec2 = [2.0, 4.0, 6.0, 8.0, 10.0]
        let cossim = cluster.cosineSimilarity(vector1: vec1, vector2: vec2)

        expect(cossim).to(beCloseTo(0.9847319278346619, within: 0.0001))
    }
    
    func testGoogleSearchClustering() async throws {
        let cluster = Cluster()
        var UUIDs: [UUID] = []
        
        for _ in 0...8 {
            UUIDs.append(UUID())
        }
        
        let pages = [
            Page(id: UUIDs[0], parentId: nil, url: URL(string: "https://www.google.com/search?q=mozart")!, title: "mozart - Google Search"),
            Page(id: UUIDs[1], parentId: nil, url: URL(string: "https://www.google.com/search?q=classical%20music%20mozart")!, title: "classical music mozart - Google Search"),
            Page(id: UUIDs[2], parentId: nil, url: URL(string: "https://www.google.com/search?q=cat")!, title: "cat - Google Search"),
            Page(id: UUIDs[3], parentId: nil, url: URL(string: "https://www.google.com/search?q=dog")!, title: "dog - Google Search"),
            Page(id: UUIDs[4], parentId: nil, url: URL(string: "https://www.google.com/search?q=worm")!, title: "worm - Google Search"),
            Page(id: UUIDs[5], parentId: nil, url: URL(string: "https://www.google.com/search?q=snake")!, title: "snake - Google Search"),
            Page(id: UUIDs[6], parentId: nil, url: URL(string: "https://www.google.com/search?q=beethoven")!, title: "beethoven - Google Search"),
            Page(id: UUIDs[7], parentId: nil, url: URL(string: "https://www.google.com/search?q=musique%20classique")!, title: "musique classique - Google Search")
        ]
        var clusteredPageIds: [[UUID]] = []
        
        for page in pages {
            clusteredPageIds = try await cluster.add(page: page).pageGroups
        }
        
        expect(clusteredPageIds.count).to(equal(5))
    }

    /// Test that scoring of textual similarity between two texts is done correctly. At the same opportunity, test all similarity matrices (entities and navigation, in addition to text)
    func testScoreTextualEmbedding() async throws {
        let cluster = Cluster()
        var UUIDs: [UUID] = []
        
        for _ in 0...4 {
            UUIDs.append(UUID())
        }
        
        let pages = [
            Page(id: UUIDs[0], parentId: nil, url: URL(string: "https://en.wikipedia.org/wiki/Roger_Federer")!, title: nil, originalContent: ["Federer has played in an era where he dominated men's tennis together with Rafael Nadal and Novak Djokovic, who have been collectively referred to as the Big Three and are widely considered three of the greatest tennis players of all-time. A Wimbledon junior champion in 1998, Federer won his first Grand Slam singles title at Wimbledon in 2003 at age 21. In 2004, he won three out of the four major singles titles and the ATP Finals, a feat he repeated in 2006 and 2007. From 2005 to 2010, Federer made 18 out of 19 major singles finals. During this span, he won his fifth consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-ups to Nadal, his main rival up until 2010. At age 27, he also surpassed Pete Sampras's then-record of 14 Grand Slam men's singles titles at Wimbledon in 2009."]),
            Page(id: UUIDs[1], parentId: UUIDs[0], url: URL(string: "https://en.wikipedia.org/wiki/Rafael_Nadal")!, title: nil, originalContent: ["From childhood through most of his professional career, Nadal was coached by his uncle Toni. He was one of the most successful teenagers in ATP Tour history, reaching No. 2 in the world and winning 16 titles before his 20th birthday, including his first French Open and six Masters events. Nadal became No. 1 for the first time in 2008 after his first major victory off clay against his rival, the longtime top-ranked Federer, in a historic Wimbledon final. He also won an Olympic gold medal in singles that year in Beijing. After defeating Djokovic in the 2010 US Open final, the 24-year-old Nadal became the youngest man in the Open Era to achieve the career Grand Slam, and also became the first man to win three majors on three different surfaces (hard, grass and clay) the same calendar year. With his Olympic gold medal, he is also one of only two male players to complete the career Golden Slam."]),
            Page(id: UUIDs[2], parentId: nil, url: URL(string: "https://fr.wikipedia.org/wiki/Roger_Federer")!, title: nil, originalContent: ["Sa victoire à Roland-Garros en 2009 lui a permis d'accomplir le Grand Chelem en carrière sur quatre surfaces différentes. En s'adjugeant ensuite l'Open d'Australie en 2010, il devient le premier joueur de l'histoire à avoir conquis l'ensemble de ses titres du Grand Chelem sur un total de cinq surfaces, depuis le remplacement du Rebound Ace australien par une nouvelle surface : le Plexicushion. Federer a réalisé le Petit Chelem de tennis à trois reprises, en 2004, 2006 et 2007, ce qui constitue à égalité avec Novak Djokovic, le record masculin toutes périodes confondues. Il est ainsi l'unique athlète à avoir gagné trois des quatre tournois du Grand Chelem deux années successives. Il atteint à trois reprises, et dans la même saison, les finales des quatre tournois majeurs, en 2006, 2007 et 2009, un fait unique dans l'histoire de ce sport."]),
            Page(id: UUIDs[3], parentId: UUIDs[2], url: URL(string: "https://fr.wikipedia.org/wiki/Rafael_Nadal")!, title: nil, originalContent: ["Il est considéré par tous les spécialistes comme le meilleur joueur sur terre battue de l'histoire du tennis, établissant en effet des records majeurs, et par la plupart d'entre eux comme l'un des meilleurs joueurs de simple de tous les temps, si ce n’est le meilleur. Il a remporté vingt tournois du Grand Chelem (un record qu'il détient avec Roger Federer et Novak Djokovic) et est le seul joueur à avoir remporté treize titres en simple dans un de ces quatre tournois majeurs : à Roland-Garros où il s'est imposé de 2005 à 2008, de 2010 à 2014, puis de 2017 à 2020. À l'issue de l'édition 2021, où il est détrôné en demi-finale par Novak Djokovic, il présente un bilan record de cent-cinq victoires pour trois défaites dans ce tournoi parisien, et ne compte aucune défaite en finale. Il a remporté également le tournoi de Wimbledon en 2008 et 2010, l'Open d'Australie 2009 et l'US Open 2010, 2013, 2017 et 2019. Il est ainsi le septième joueur de l'histoire du tennis à réaliser le « Grand Chelem en carrière » en simple. À ce titre, Rafael Nadal est le troisième joueur et le plus jeune à s'être imposé durant l'ère Open dans les quatre tournois majeurs sur quatre surfaces différentes, performance que seuls Roger Federer, Andre Agassi et Novak Djokovic ont accomplie."]),
            Page(id: UUIDs[4], parentId: nil, url: URL(string: "https://www.youtube.com")!, title:nil, originalContent: ["All"])
        ]
        var clusteredPageIds: [[UUID]] = []
        
        for page in pages {
            clusteredPageIds = try await cluster.add(page: page).pageGroups
        }
        
        expect(clusteredPageIds.count).to(equal(2))
    }
    
    func testMixPageNote() async throws {
        let cluster = Cluster()
        var pageUUIDs: [UUID] = []
        var noteUUIDs: [UUID] = []
        
        for _ in 0...2 {
            pageUUIDs.append(UUID())
            noteUUIDs.append(UUID())
        }
        
        let pages = [
            Page(id: pageUUIDs[0], parentId: nil, url: URL(string: "https://en.wikipedia.org/wiki/Roger_Federer")!, title: nil, originalContent: ["Federer has played in an era where he dominated men's tennis together with Rafael Nadal and Novak Djokovic, who have been collectively referred to as the Big Three and are widely considered three of the greatest tennis players of all-time. A Wimbledon junior champion in 1998, Federer won his first Grand Slam singles title at Wimbledon in 2003 at age 21. In 2004, he won three out of the four major singles titles and the ATP Finals, a feat he repeated in 2006 and 2007. From 2005 to 2010, Federer made 18 out of 19 major singles finals. During this span, he won his fifth consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-ups to Nadal, his main rival up until 2010. At age 27, he also surpassed Pete Sampras's then-record of 14 Grand Slam men's singles titles at Wimbledon in 2009."]),
            Page(id: pageUUIDs[1], parentId: nil, url: URL(string: "https://en.wikipedia.org/wiki/Rafael_Nadal")!, title: nil, originalContent: ["From childhood through most of his professional career, Nadal was coached by his uncle Toni. He was one of the most successful teenagers in ATP Tour history, reaching No. 2 in the world and winning 16 titles before his 20th birthday, including his first French Open and six Masters events. Nadal became No. 1 for the first time in 2008 after his first major victory off clay against his rival, the longtime top-ranked Federer, in a historic Wimbledon final. He also won an Olympic gold medal in singles that year in Beijing. After defeating Djokovic in the 2010 US Open final, the 24-year-old Nadal became the youngest man in the Open Era to achieve the career Grand Slam, and also became the first man to win three majors on three different surfaces (hard, grass and clay) the same calendar year. With his Olympic gold medal, he is also one of only two male players to complete the career Golden Slam."]),
            Page(id: pageUUIDs[2], parentId: nil, url: URL(string: "https://www.youtube.com")!, title:nil, originalContent: ["All"])
        ]
        
        let notes = [
            ClusteringNote(id: noteUUIDs[0], title: "Roger Federer", content: ["Sa victoire à Roland-Garros en 2009 lui a permis d'accomplir le Grand Chelem en carrière sur quatre surfaces différentes. En s'adjugeant ensuite l'Open d'Australie en 2010, il devient le premier joueur de l'histoire à avoir conquis l'ensemble de ses titres du Grand Chelem sur un total de cinq surfaces, depuis le remplacement du Rebound Ace australien par une nouvelle surface : le Plexicushion. Federer a réalisé le Petit Chelem de tennis à trois reprises, en 2004, 2006 et 2007, ce qui constitue à égalité avec Novak Djokovic, le record masculin toutes périodes confondues. Il est ainsi l'unique athlète à avoir gagné trois des quatre tournois du Grand Chelem deux années successives. Il atteint à trois reprises, et dans la même saison, les finales des quatre tournois majeurs, en 2006, 2007 et 2009, un fait unique dans l'histoire de ce sport."]),
            ClusteringNote(id: noteUUIDs[1], title: "Rafael Nadal", content: ["Il est considéré par tous les spécialistes comme le meilleur joueur sur terre battue de l'histoire du tennis, établissant en effet des records majeurs, et par la plupart d'entre eux comme l'un des meilleurs joueurs de simple de tous les temps, si ce n’est le meilleur. Il a remporté vingt tournois du Grand Chelem (un record qu'il détient avec Roger Federer et Novak Djokovic) et est le seul joueur à avoir remporté treize titres en simple dans un de ces quatre tournois majeurs : à Roland-Garros où il s'est imposé de 2005 à 2008, de 2010 à 2014, puis de 2017 à 2020. À l'issue de l'édition 2021, où il est détrôné en demi-finale par Novak Djokovic, il présente un bilan record de cent-cinq victoires pour trois défaites dans ce tournoi parisien, et ne compte aucune défaite en finale. Il a remporté également le tournoi de Wimbledon en 2008 et 2010, l'Open d'Australie 2009 et l'US Open 2010, 2013, 2017 et 2019. Il est ainsi le septième joueur de l'histoire du tennis à réaliser le « Grand Chelem en carrière » en simple. À ce titre, Rafael Nadal est le troisième joueur et le plus jeune à s'être imposé durant l'ère Open dans les quatre tournois majeurs sur quatre surfaces différentes, performance que seuls Roger Federer, Andre Agassi et Novak Djokovic ont accomplie."])
        ]
        
        var clusteredPageIds: [[UUID]] = []
        var clusteredNoteIds: [[UUID]] = []
        
        for page in pages {
            clusteredPageIds = try await cluster.add(page: page).pageGroups
        }
        
        for note in notes {
            clusteredNoteIds = try await cluster.add(note: note).noteGroups
        }
        
        expect(clusteredPageIds.count).to(equal(2))
        expect(clusteredNoteIds.count).to(equal(3))
    }

    /// Test that similarities between notes/active-sources to suggestions are returned correctly, for the sake of correct scoring of suggestions
    func testCreateSimilarities() throws {
        let cluster = Cluster()
        
        cluster.textualSimilarityMatrix.matrix = Matrix([[0, 0, 0, 0.9, 0.8, 0.7],
                                                         [0, 0, 0, 0.5, 0.5, 0.5],
                                                         [0, 0, 0, 0.1, 0.2, 0.3],
                                                         [0.9, 0.5, 0.1, 0, 0.5, 0.2],
                                                         [0.8, 0.5, 0.2, 0.5, 0, 0.3],
                                                         [0.7, 0.5, 0.3, 0.2, 0.3, 0]])
        cluster.notes = [ClusteringNote(id: UUID(), title: "First note", content: ["note"]),
                         ClusteringNote(id: UUID(), title: "Second note", content: ["note"]),
                         ClusteringNote(id: UUID(), title: "Third note", content: ["note"])
        ]
        cluster.pages = [Page(id: UUID(), parentId: nil, title: "First page", cleanedContent: "page"),
                         Page(id: UUID(), parentId: nil, title: "Second page", cleanedContent: "page"),
                         Page(id: UUID(), parentId: nil, title: "Third page", cleanedContent: "page")
        ]
        
        let noteGroups = [[cluster.notes[0].id], [cluster.notes[1].id], [cluster.notes[2].id], [], []]
        let pageGroups = [[], [cluster.pages[2].id], [], [cluster.pages[0].id, cluster.pages[1].id]]
        let mySimilarities = cluster.createSimilarities(pageGroups: pageGroups, noteGroups: noteGroups)
        
        expect(mySimilarities[cluster.notes[0].id]) == [:]
        expect(mySimilarities[cluster.notes[1].id]) == [cluster.pages[2].id: 0.5]
        expect(mySimilarities[cluster.notes[2].id]) == [:]
        expect(mySimilarities[cluster.pages[0].id]) == [cluster.pages[1].id: 0.5]
    }
    
    /// Test for the getSubmatrix method
    func testGetSubMatrix() throws {
        let cluster = Cluster()
        let myMatrix = Matrix([[0, 1, 2, 3],
                              [4, 5, 6, 7],
                              [8, 9, 10, 11],
                              [12, 13, 14, 15]])
        
        expect(try cluster.getSubmatrix(of: myMatrix, withIndeces:[0, 2]).flat) == [0.0, 2.0, 8.0, 10.0]
    }
    
    /// Test that two notes alone in a subgroup are separated correctly
    func testSeparateTwoNotesOnly() throws {
        let cluster = Cluster()
        let adjacencyOnlyNotes = Matrix([[0, -1],
                                        [-1, 0]])
        let result = try cluster.spectralClustering(on: adjacencyOnlyNotes, numGroups: 2, numNotes: 2)
        
        expect(Set(result)) == Set([0, 1])
    }
    
    /// Test that two notes in a group with pages are separated correctly
    func testSeparateNotesWithPages() throws {
        let cluster = Cluster()
        let adjacencySubgroup = Matrix([[0, -1, 0.5, 0.5],
                                        [-1, 0, 1, 0],
                                        [0.5, 1, 0, 0.2],
                                        [0.5, 0, 0.2, 0]])
        let result = try cluster.spectralClustering(on: adjacencySubgroup, numGroups: 2, numNotes: 2)
        
        expect(Set(result)) == Set([0, 1])
        expect(result[0]) != result[1]
    }
    
    func testRemoveNote() async throws {
        let cluster = Cluster()
        var noteUUIDs: [UUID] = []
        
        for i in 0...2 {
            noteUUIDs.append(UUID())
            let myNote = ClusteringNote(id: noteUUIDs[i], title: "My note", content: ["This is my note"])
            _ = try await cluster.add(note: myNote)
            
        }
        
        expect(cluster.notes.count) == 3
        expect(cluster.notes[0].id) == noteUUIDs[0]
        expect(cluster.notes[1].id) == noteUUIDs[1]
        expect(cluster.notes[2].id) == noteUUIDs[2]
        
        try cluster.removeNote(noteId: noteUUIDs[0])
        
        expect(cluster.notes.count) == 2
        expect(cluster.notes[0].id) == noteUUIDs[1]
        expect(cluster.notes[1].id) == noteUUIDs[2]
    }
    
    func testRemovePage() async throws {
        let cluster = Cluster()
        // Here we don't want to test that notes with little content are not added
        var pageUUIDs: [UUID] = []
        
        for i in 0...2 {
            pageUUIDs.append(UUID())
            let myPage = Page(id: pageUUIDs[i], title: "My note", originalContent: ["This is my note"])
            _ = try await cluster.add(page: myPage)
            
        }
        
        expect(cluster.pages.count) == 3
        expect(cluster.pages[0].id) == pageUUIDs[0]
        expect(cluster.pages[1].id) == pageUUIDs[1]
        expect(cluster.pages[2].id) == pageUUIDs[2]
        
        try cluster.removePage(pageId: pageUUIDs[0])
        
        expect(cluster.pages.count) == 2
        expect(cluster.pages[0].id) == pageUUIDs[1]
        expect(cluster.pages[1].id) == pageUUIDs[2]
    }
    
    func testGetExportInformationForIdPage() async throws {
        let cluster = Cluster()
        let pageId = UUID()
        let myPage = Page(id: pageId, parentId: nil, title: "Roger Federer", cleanedContent: "He was born on 8 August 1981 in Basel.")
        
        _ = try await cluster.add(page: myPage)
        
        let pageInformation = cluster.getExportInformationForId(id: pageId)
        let expectedInformation = InformationForId(title: "Roger Federer", cleanedContent: "He was born on 8 August 1981 in Basel.")
        
        expect(pageInformation) == expectedInformation
        
        let emptyInformation = cluster.getExportInformationForId(id: UUID())
        
        expect(emptyInformation) == InformationForId()
    }

    func testGetInformationForIdNote() async throws {
        let cluster = Cluster()
        let noteId = UUID()
        let myNote = ClusteringNote(id: noteId, title: "Roger Federer", content: ["Federer has played in an era where he dominated men's tennis along with Rafael Nadal and Novak Djokovic. Referred to as the Big Three, they are considered by some to be the three greatest tennis players of all time.[c] A Wimbledon junior champion in 1998, Federer won his first major singles title at Wimbledon in 2003 at age 21. In 2004, he won three of the four major singles titles and the ATP Finals,[d] a feat he repeated in 2006 and 2007. From 2005 to 2010, he made 18 out of 19 major singles finals. During this span, he won five consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-up finishes to Nadal, his main rival until 2010. At age 27, he surpassed Pete Sampras's record of 14 major men's singles titles at Wimbledon in 2009."])
        
        _ = try await cluster.add(note: myNote)
        
        let noteInformation = cluster.getExportInformationForId(id: noteId)
        let expectedInformation = InformationForId(title: "Roger Federer", cleanedContent: "Federer has played in an era where he dominated men\'s tennis along with Rafael Nadal and Novak Djokovic. Referred to as the Big Three, they are considered by some to be the three greatest tennis players of all time.[c] A Wimbledon junior champion in 1998, Federer won his first major singles title at Wimbledon in 2003 at age 21. In 2004, he won three of the four major singles titles and the ATP Finals,[d] a feat he repeated in 2006 and 2007. From 2005 to 2010, he made 18 out of 19 major singles finals. During this span, he won five consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-up finishes to Nadal, his main rival until 2010. At age 27, he surpassed Pete Sampras\'s record of 14 major men\'s singles titles at Wimbledon in 2009.")
        
        expect(noteInformation) == expectedInformation
        
        let emptyInformation = cluster.getExportInformationForId(id: UUID())
        
        expect(emptyInformation) == InformationForId()
    }
    
    // swiftlint:disable:next file_length
}
