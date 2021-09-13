import Nimble
import XCTest
import LASwift
import NaturalLanguage
@testable import Clustering

// swiftlint:disable:next type_body_length
class ClusteringTests: XCTestCase {

    /// Test that initialization of the clustering module is done with all parameters as expected
    func testInitialization() throws {
        let cluster = Cluster()
        expect(cluster.candidate) == 2
        expect(cluster.laplacianCandidate) == .randomWalkLaplacian
        expect(cluster.matrixCandidate) == .combinationSigmoidWithTextErasure
        expect(cluster.noteMatrixCandidate) == .fixed
        expect(cluster.numClustersCandidate) == .biggestDistanceInPercentages
        expect(cluster.weights[.navigation]) == 0.5
        expect(cluster.weights[.text]) == 0.8
        expect(cluster.weights[.entities]) == 0.5
    }

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

    /// Test removing of data points from the navigation matrix. When a data point is removed, that is connected to different other data points, connections between these data points are automatically created
    func testRemovalFromNavigationMatrix() throws {
        let cluster = Cluster()
        cluster.navigationMatrix.matrix = Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        try cluster.navigationMatrix.removeDataPoint(index: 1)
        expect(cluster.navigationMatrix.matrix.flat) == [0, 1, 1, 0]
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

    /// Test that the language detection method works properly (MacOS 11 and onwards only)
    func testLanguageDetection() throws {
        if #available(iOS 14, macOS 11, *) {
            let cluster = Cluster()
            expect(cluster.getTextLanguage(text: "Roger Federer is the greatest of all time")) == NLLanguage.english
            expect(cluster.getTextLanguage(text: "Roger Federer est le meilleur joueur de tous les temps")) == NLLanguage.french
            expect(cluster.getTextLanguage(text: "רוג׳ר פדרר הוא השחקן הטוב ביותר בכל הזמנים")) == NLLanguage.hebrew
        }
    }

    /// Test that scoring of textual similarity between two texts is done correctly. At the same opportunity, test that no entities are detected when they shouldn't be. Non-English content is also tested
    func noTgestScoreTextualEmbedding() throws {
        if #available(iOS 14, macOS 11, *) {
            let cluster = Cluster()
            let pages = [
                Page(id: 0, parentId: nil, title: nil, content: "A man is eating food."),
                Page(id: 1, parentId: 0, title: nil, content: "A man is eating a piece of bread."),
                Page(id: 2, parentId: 0, title: nil, content: "Un homme mange du pain et evidemment il faut plus de text pour reconnaître le français."),
                Page(id: 3, parentId: nil, title: nil, content: "Un homme mange mais je vais ajouter un peu plus de texte quand même.")
                ]
            let expectation = self.expectation(description: "Add page expectation")
            for page in pages.enumerated() {
                cluster.add(page: page.element, ranking: nil, completion: { result in
                    switch result {
                    case .failure(let error):
                        XCTFail(error.localizedDescription)
                    case .success(let result):
                        _ = result.0
                    }
                    if page.offset == pages.count - 1 {
                        expectation.fulfill()
                    }
                })
            }
            wait(for: [expectation], timeout: 1)
            expect(cluster.pages[2].language) == NLLanguage.french
            expect(cluster.pages[3].language) == NLLanguage.french
            expect(cluster.textualSimilarityMatrix.matrix.flat).to(beCloseTo([0, 0.8294351697354525, 1, 1, 0.8294351697354525, 0, 1, 1, 1, 1, 0, 0.9531, 1, 1, 0.9531, 0], within: 0.0001))
            expect(cluster.entitiesMatrix.matrix.flat).to(beCloseTo([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], within: 0.0001))
        }
    }

    /// Test the whole process of starting a session, adding pages and clustering, when only a  navigation matrix is available.
    func testClusteringWithOnlyNavigation() throws {
        let cluster = Cluster()
        let ids: [UInt64] = Array(0...5)
        let parents = [1: 0, 2: 0, 4: 3, 5: 1]
        let correct_results = [[[ids[0]]], [[ids[0], ids[1]]], [[ids[0], ids[1], ids[2]]], [[ids[0], ids[1], ids[2]], [ids[3]]], [[ids[0], ids[1], ids[2]], [ids[3], ids[4]]], [[ids[0], ids[1], ids[2], ids[5]], [ids[3], ids[4]]]]
        let expectation = XCTestExpectation(description: "Add page expectation")
        for i in 0...5 {
            var from: UInt64?
            if let parent = parents[i] {
                from = ids[parent]
            }
            let page = Page(id: ids[i], parentId: from, title: nil, content: nil)
            cluster.add(page: page, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    XCTFail(error.localizedDescription)
                case .success(let result):
                    expect(result.0) == correct_results[i]
                }
                if i == 5 {
                    expectation.fulfill()
                }
            })
        }
        wait(for: [expectation], timeout: 1)
    }

    /// Test that entities are detected correctly in a text
    func testFindingEntities() throws {
        let cluster = Cluster()
        let myText = "Roger Federer is the best tennis player to ever play the game, but Rafael Nadal is best on clay"
        let myEntities = cluster.findEntitiesInText(text: myText)
        expect(myEntities.entities["PlaceName"]) == []
        expect(myEntities.entities["OrganizationName"]) == []
        expect(myEntities.entities["PersonalName"]) == ["roger federer", "rafael nadal"]
    }

    /// Test that the Jaccard similarity measure between two sets of entities is computed correctly
    func testJaccardSimilarityMeasure() throws {
        let cluster = Cluster()
        let firstText = "Roger Federer is the best tennis player to ever play the game, but Rafael Nadal is best on clay"
        let secondText = "Rafael Nadal won Roland Garros 13 times"
        let firstTextEntities = cluster.findEntitiesInText(text: firstText)
        let secondTextEntities = cluster.findEntitiesInText(text: secondText)
        let similarity = cluster.jaccardEntities(entitiesText1: firstTextEntities, entitiesText2: secondTextEntities)
        expect(similarity).to(beCloseTo(0.5, within: 0.0001))
    }

    /// Test that titles are taken into account correctly for the sake of entity comparison
    func testEntitySimilarityOverTitles() throws {
        let cluster = Cluster()
        let expectation = self.expectation(description: "Add page expectation")
        let pages = [
            Page(id: 0, parentId: nil, title: "roger federer - Google search", content: nil),
            Page(id: 1, parentId: 0, title: "Roger Federer", content: nil),
            Page(id: 2, parentId: 0, title: "Pete Sampras", content: nil)
            ]
        for page in pages.enumerated() {
            cluster.add(page: page.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    XCTFail(error.localizedDescription)
                case .success(let result):
                    _ = result.0
                }
                if page.offset == pages.count - 1 {
                    expectation.fulfill()
                }
            })
        }
        wait(for: [expectation], timeout: 1)

        let expectedEntitiesMatrix = [0.0, 1.0, 0.0,
                                      1.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0]
        expect(cluster.entitiesMatrix.matrix.flat).to(beCloseTo(expectedEntitiesMatrix, within: 0.0001))
    }

    /// Test all similarity matrices
    func testAllSimilarityMatrices() throws {
        let cluster = Cluster()
        let expectation = self.expectation(description: "Add page expectation")
        let pages = [
            Page(id: 0, parentId: nil, title: nil, content: "Roger Federer is the best tennis player to ever play the game, but Rafael Nadal is best on clay"),
            Page(id: 1, parentId: 0, title: nil, content: "Tennis is a very fun game"),
            Page(id: 2, parentId: 0, title: nil, content: "Pete Sampras and Roger Federer played 4 exhibition matches in 2008")
            ]
        for page in pages.enumerated() {
            cluster.add(page: page.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    XCTFail(error.localizedDescription)
                case .success(let result):
                    _ = result.0
                }
                if page.offset == pages.count - 1 {
                    expectation.fulfill()
                }
            })
        }
        wait(for: [expectation], timeout: 1)

        let expectedEntitiesMatrix = [0.0, 0.0, 0.5,
                                    0.0, 0.0, 0.0,
                                    0.5, 0.0, 0.0]
        let expectedNavigationMatrix = [0.0, 1.0, 1.0,
                                        1.0, 0.0, 0.0,
                                        1.0, 0.0, 0.0]
        let expectedTextualMatrix = [0.0, 0.46988745662121795, 0.3628206314147012,
                                    0.46988745662121795, 0.0, 0.19001699954776027,
                                    0.3628206314147012, 0.19001699954776027, 0.0]
        if #available(iOS 14, macOS 11, *) {
            expect(cluster.pages[0].language) == NLLanguage.english
            expect(cluster.pages[1].language) == NLLanguage.english
            expect(cluster.pages[2].language) == NLLanguage.english
            expect(cluster.textualSimilarityMatrix.matrix.flat).to(beCloseTo(expectedTextualMatrix, within: 0.0001))
        }
        expect(cluster.entitiesMatrix.matrix.flat).to(beCloseTo(expectedEntitiesMatrix, within: 0.0001))
        expect(cluster.navigationMatrix.matrix.flat).to(beCloseTo(expectedNavigationMatrix, within: 0.0001))
    }

    /// Test that the 'add' function sends back the sendRanking flag when the clustering process exceeds the time threshold
    func testRaiseRemoveFlag() throws {
        let cluster = Cluster()
        let expectation = self.expectation(description: "Raise remove flag")
        cluster.timeToRemove = 0.0
        let pages = [
            Page(id: 0, parentId: nil, title: nil, content: "Roger Federer is the best tennis player to ever play the game, but Rafael Nadal is best on clay"),
            Page(id: 1, parentId: 0, title: nil, content: "Tennis is a very fun game"),
            Page(id: 2, parentId: 0, title: nil, content: "Pete Sampras and Roger Federer played 4 exhibition matches in 2008")
            ]
        for page in pages.enumerated() {
            cluster.add(page: page.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    XCTFail(error.localizedDescription)
                case .success(let result):
                    if page.offset == pages.count - 1 {
                        expect(result.sendRanking) == true
                    }
                }
                if page.offset == pages.count - 1 {
                    expectation.fulfill()
                }
            })
        }
        wait(for: [expectation], timeout: 1)
    }

    /// Test that when a ranking is sent along with  a request to 'add', the 3 least ranked pages are removed
    func testPageRemoval() throws {
        let cluster = Cluster()
        let expectation = self.expectation(description: "Add page expectation")
        let pages = [
            Page(id: 0, parentId: nil, title: "man", content: "A man is eating food."),
            Page(id: 1, parentId: 0, title: "girl", content: "The girl is carrying a baby."),
            Page(id: 2, parentId: 0, title: "man", content: "A man is eating food."),
            Page(id: 3, parentId: 0, title: "girl", content: "The girl is carrying a baby."),
            Page(id: 4, parentId: 0, title: "girl", content: "The girl is carrying a baby."),
            Page(id: 5, parentId: 0, title: "man", content: "A man is eating food."),
            Page(id: 6, parentId: 0, title: "fille", content: "La fille est en train de porter un bébé.")
            ]
        for page in pages.enumerated() {
            var ranking: [UInt64]?
            if page.offset == pages.count - 1 {
                ranking = [1, 4, 2, 3, 5, 0]
            }
            cluster.add(page: page.element, ranking: ranking, completion: { result in
                switch result {
                case .failure(let error):
                    XCTFail(error.localizedDescription)
                case .success(let result):
                    _ = result
                }
                if page.offset == pages.count - 1 {
                    expectation.fulfill()
                }
            })
            if page.offset == 4 {
                let myUUID = UUID()
                let myNote = ClusteringNote(id: myUUID, title: "Roger Federer", content: "Roger Federer is the best Tennis player in history")
                cluster.add(note: myNote, ranking: nil, completion: { result in
                    switch result {
                    case .failure(let error):
                        XCTFail(error.localizedDescription)
                    case .success(let result):
                        _ = result
                    }
                })
            }
        }
        wait(for: [expectation], timeout: 1)
        var attachedPages = [UInt64]()
        for page in cluster.pages {
            attachedPages += page.attachedPages
        }
        expect(Set(attachedPages)) == Set([1, 4, 2])
        expect(cluster.adjacencyMatrix.rows) == 5 // 4 pages and one note
        expect(cluster.pages.count) == 4
        expect(cluster.notes.count) == 1
    }

    /// A page that was removed from the matrices is visited again by the user. Test that it is readded correctly and removed from attachedPages
    func testRevisitPageAfterRemoval() throws {
        let cluster = Cluster()
        let firstExpectation = self.expectation(description: "Add page expectation")
        let secondExpectation = self.expectation(description: "Add page expectation")
        let firstPages = [
            Page(id: 0, parentId: nil, title: "Page 1", content: "A man is eating food."),
            Page(id: 1, parentId: 0, title: "Page 2", content: "The girl is carrying a baby."),
            Page(id: 2, parentId: 0, title: "Page 3", content: "A man is eating food.")
            ]
        let secondPages = [
            Page(id: 3, parentId: 0, title: "Page 4", content: "The girl is carrying a baby."),
            Page(id: 4, parentId: 0, title: "Page 5", content: "The girl is carrying a baby.")
            ]
        for page in firstPages.enumerated() {
            cluster.add(page: page.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    XCTFail(error.localizedDescription)
                case .success(let result):
                    _ = result.0
                }
                if page.offset == firstPages.count - 1 {
                    firstExpectation.fulfill()
                }
            })
        }
        wait(for: [firstExpectation], timeout: 1)
        cluster.pages[0].attachedPages = [3]
        cluster.pages[1].attachedPages = [4]
        for page in secondPages.enumerated() {
            cluster.add(page: page.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    XCTFail(error.localizedDescription)
                case .success(let result):
                    _ = result.0
                }
                if page.offset == secondPages.count - 1 {
                    secondExpectation.fulfill()
                }
            })
        }
        wait(for: [secondExpectation], timeout: 1)
        expect(cluster.pages[0].attachedPages) == []
        expect(cluster.pages[1].attachedPages) == []
    }

    /// When removing a page from the matrix, chage that if the most similar data point to that page is a note, that does not create a problem
    func testRemovingPageWithSimilarNote() throws {
        let cluster = Cluster()
        let expectation = self.expectation(description: "Add note expectation")
        for i in 0...5 {
            let myPage = Page(id: UInt64(i), parentId: nil, title: nil, content: "Here's some text for you")
            // The pages themselves don't matter as we will later force the similarity matrix
            cluster.add(page: myPage, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    XCTFail(error.localizedDescription)
                case .success(let result):
                    _ = result.0
                }
            })
        }
        for i in 0...2 {
            let myNote = ClusteringNote(id: UUID(), title: "My note", content: "This is my note")
            cluster.add(note: myNote, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    XCTFail(error.localizedDescription)
                case .success(let result):
                    _ = result.0
                }
                if i == 2 {
                    expectation.fulfill()
                }
            })
        }
        wait(for: [expectation], timeout: 1)
        cluster.adjacencyMatrix = Matrix([[0, 0, 0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                                   [0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                   [0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                                   [0.9, 0.1, 0.3, 0, 0.5, 0.5, 0.5, 0.5, 0.5],
                                   [0.8, 0.2, 0.3, 0.5, 0, 0.7, 0.2, 0.1, 0.1],
                                   [0.7, 0.3, 0.3, 0.5, 0.7, 0, 0.6, 0.2, 0.1],
                                   [0.6, 0.4, 0.3, 0.5, 0.2, 0.6, 0, 0.9, 0.3],
                                   [0.5, 0.5, 0.3, 0.5, 0.1, 0.2, 0.9, 0, 0.4],
                                   [0.4, 0.6, 0.3, 0.5, 0.1, 0.1, 0.3, 0.4, 0]])
        try cluster.remove(ranking: [0])
        expect(cluster.pages[0].id) == UInt64(1)
        expect(cluster.pages[0].attachedPages) == [0]
    }
    // swiftlint:disable:next file_length
}
