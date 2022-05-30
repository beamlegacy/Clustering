import Nimble
import XCTest
import LASwift
import NaturalLanguage
@testable import Clustering
import Accelerate
import Clustering

// swiftlint:disable:next type_body_length
class ClusteringTests: XCTestCase {

    /// Test that initialization of the clustering module is done with all parameters as expected
    func testInitialization() throws {
        let cluster = Cluster()
        expect(cluster.candidate) == 2
        expect(cluster.laplacianCandidate) == .randomWalkLaplacian
        expect(cluster.matrixCandidate) == .combinationSigmoidWithTextErasure
        expect(cluster.noteMatrixCandidate) == .sigmoidOnEntities
        expect(cluster.numClustersCandidate) == .biggestDistanceInPercentages
        expect(cluster.weights[.navigation]) == 0.5
        expect(cluster.weights[.text]) == 0.9
        expect(cluster.weights[.entities]) == 0.2
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

    /// Test that scoring of textual similarity between two texts is done correctly. At the same opportunity, test all similarity matrices (entities and navigation, in addition to text)
    func testScoreTextualEmbedding() throws {
        if #available(iOS 14, macOS 11, *) {
            let cluster = Cluster()
            var UUIDs: [UUID] = []
            for _ in 0...4 {
                UUIDs.append(UUID())
            }
            let pages = [
                Page(id: UUIDs[0], parentId: nil, url: URL(string: "https://en.wikipedia.org/wiki/Roger_Federer")!, title: nil, originalContent: ["Federer has played in an era where he dominated men's tennis together with Rafael Nadal and Novak Djokovic, who have been collectively referred to as the Big Three and are widely considered three of the greatest tennis players of all-time.[c] A Wimbledon junior champion in 1998, Federer won his first Grand Slam singles title at Wimbledon in 2003 at age 21. In 2004, he won three out of the four major singles titles and the ATP Finals,[d] a feat he repeated in 2006 and 2007. From 2005 to 2010, Federer made 18 out of 19 major singles finals. During this span, he won his fifth consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-ups to Nadal, his main rival up until 2010. At age 27, he also surpassed Pete Sampras's then-record of 14 Grand Slam men's singles titles at Wimbledon in 2009."]),
                Page(id: UUIDs[1], parentId: UUIDs[0], url: URL(string: "https://en.wikipedia.org/wiki/Rafael_Nadal")!, title: nil, originalContent: ["From childhood through most of his professional career, Nadal was coached by his uncle Toni. He was one of the most successful teenagers in ATP Tour history, reaching No. 2 in the world and winning 16 titles before his 20th birthday, including his first French Open and six Masters events. Nadal became No. 1 for the first time in 2008 after his first major victory off clay against his rival, the longtime top-ranked Federer, in a historic Wimbledon final. He also won an Olympic gold medal in singles that year in Beijing. After defeating Djokovic in the 2010 US Open final, the 24-year-old Nadal became the youngest man in the Open Era to achieve the career Grand Slam, and also became the first man to win three majors on three different surfaces (hard, grass and clay) the same calendar year. With his Olympic gold medal, he is also one of only two male players to complete the career Golden Slam."]),
                Page(id: UUIDs[2], parentId: nil, url: URL(string: "https://fr.wikipedia.org/wiki/Roger_Federer")!, title: nil, originalContent: ["Sa victoire à Roland-Garros en 2009 lui a permis d'accomplir le Grand Chelem en carrière sur quatre surfaces différentes. En s'adjugeant ensuite l'Open d'Australie en 2010, il devient le premier joueur de l'histoire à avoir conquis l'ensemble de ses titres du Grand Chelem sur un total de cinq surfaces, depuis le remplacement du Rebound Ace australien par une nouvelle surface : le Plexicushion. Federer a réalisé le Petit Chelem de tennis à trois reprises, en 2004, 2006 et 2007, ce qui constitue à égalité avec Novak Djokovic, le record masculin toutes périodes confondues. Il est ainsi l'unique athlète à avoir gagné trois des quatre tournois du Grand Chelem deux années successives. Il atteint à trois reprises, et dans la même saison, les finales des quatre tournois majeurs, en 2006, 2007 et 2009, un fait unique dans l'histoire de ce sport."]),
                Page(id: UUIDs[3], parentId: UUIDs[2], url: URL(string: "https://fr.wikipedia.org/wiki/Rafael_Nadal")!, title: nil, originalContent: ["Il est considéré par tous les spécialistes comme le meilleur joueur sur terre battue de l'histoire du tennis, établissant en effet des records majeurs, et par la plupart d'entre eux comme l'un des meilleurs joueurs de simple de tous les temps, si ce n’est le meilleur4,5,6,7. Il a remporté vingt tournois du Grand Chelem (un record qu'il détient avec Roger Federer et Novak Djokovic) et est le seul joueur à avoir remporté treize titres en simple dans un de ces quatre tournois majeurs : à Roland-Garros où il s'est imposé de 2005 à 2008, de 2010 à 2014, puis de 2017 à 2020. À l'issue de l'édition 2021, où il est détrôné en demi-finale par Novak Djokovic, il présente un bilan record de cent-cinq victoires pour trois défaites dans ce tournoi parisien, et ne compte aucune défaite en finale. Il a remporté également le tournoi de Wimbledon en 2008 et 2010, l'Open d'Australie 2009 et l'US Open 2010, 2013, 2017 et 2019. Il est ainsi le septième joueur de l'histoire du tennis à réaliser le « Grand Chelem en carrière » en simple. À ce titre, Rafael Nadal est le troisième joueur et le plus jeune à s'être imposé durant l'ère Open dans les quatre tournois majeurs sur quatre surfaces différentes, performance que seuls Roger Federer, Andre Agassi et Novak Djokovic ont accomplie."]),
                Page(id: UUIDs[4], parentId: nil, url: URL(string: "https://www.youtube.com")!, title:nil, originalContent: ["All"])
                ]
            let expectation = self.expectation(description: "Add page expectation")
            for page in pages.enumerated() {
                cluster.add(page: page.element, ranking: nil, completion: { result in
                    switch result {
                    case .failure(let error):
                        if error as! Cluster.AdditionError != .skippingToNextAddition {
                            XCTFail(error.localizedDescription)
                        }
                    case .success(let result):
                        _ = result.0
                    }
                    if page.offset == pages.count - 1 {
                        expectation.fulfill()
                    }
                })
            }
            wait(for: [expectation], timeout: 10)
            expect(cluster.navigationMatrix.matrix.flat).to(beCloseTo([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
            /*let embedders = (NLEmbedding.sentenceEmbedding(for: NLLanguage.english), NLEmbedding.sentenceEmbedding(for: NLLanguage.french))
                      
            if embedders == (nil, nil) {
                expect(cluster.entitiesMatrix.matrix.flat).to(beCloseTo([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], within: 0.0001))
                expect(cluster.textualSimilarityMatrix.matrix.flat).to(beCloseTo([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], within: 0.0001))
            } else if embedders.1 == nil {
                expect(cluster.entitiesMatrix.matrix.flat).to(beCloseTo([0, 0.2963, 0, 0, 0, 0.2963, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], within: 0.0001))
                expect(cluster.textualSimilarityMatrix.matrix.flat).to(beCloseTo([0, 0.9201, 0, 0, 0, 0.9201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], within: 0.0001))
            } else if embedders.0 == nil {
                expect(cluster.entitiesMatrix.matrix.flat).to(beCloseTo([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0], within: 0.0001))
                expect(cluster.textualSimilarityMatrix.matrix.flat).to(beCloseTo([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9922, 0, 0, 0, 0.9922, 0, 0, 0, 0, 0, 0, 0], within: 0.0001))
            } else {*/
                expect(cluster.entitiesMatrix.matrix.flat).to(beCloseTo([0, 0.2963, 0.1667, 0.25, 0, 0.2963, 0, 0.2722, 0, 0, 0.1667, 0.2722, 0, 0.25, 0, 0.25, 0, 0.25, 0, 0, 0, 0, 0, 0, 0], within: 0.0001))
                expect(cluster.textualSimilarityMatrix.matrix.flat).to(beCloseTo([0, 0.7923, 0.6091, 0.7965, 0.0653, 0.7923, 0, 0.5650, 0.7612, -0.0117, 0.6091, 0.5650, 0, 0.6040, 0.0507, 0.7965, 0.7612, 0.6040, 0, -0.0156, 0.0653, -0.0117, 0.0507, -0.0156, 0], within: 0.0001))
            //}
        }
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
        var UUIDs: [UUID] = []
        for _ in 0...2 {
            UUIDs.append(UUID())
        }
        let pages = [
            Page(id: UUIDs[0], parentId: nil, url: URL(string: "https://www.google.com/search?q=roger%20federer&client=safari")!, title: "roger federer - Google search", cleanedContent: nil),
            Page(id: UUIDs[1], parentId: UUIDs[0], url: URL(string: "https://en.wikipedia.org/wiki/Roger_Federer")!, title: "Roger Federer", cleanedContent: nil),
            Page(id: UUIDs[2], parentId: UUIDs[0], url: URL(string: "https://en.wikipedia.org/wiki/Pete_Sampras")!, title: "Pete Sampras", cleanedContent: nil)
            ]
        for page in pages.enumerated() {
            cluster.add(page: page.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    if error as! Cluster.AdditionError != Cluster.AdditionError.skippingToNextAddition {
                        XCTFail(error.localizedDescription)
                    }
                case .success(let result):
                    _ = result.0
                }
                if page.offset == pages.count - 1 {
                    expectation.fulfill()
                }
            })
        }
        wait(for: [expectation], timeout: 10)

        let expectedEntitiesMatrix = [0.0, 1.0, 0.0,
                                      1.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0]
        expect(cluster.entitiesMatrix.matrix.flat).to(beCloseTo(expectedEntitiesMatrix, within: 0.0001))
    }

    /// Test that the 'add' function sends back the sendRanking flag when the clustering process exceeds the time threshold
    func testRaiseRemoveFlag() throws {
        let cluster = Cluster()
        let expectation = self.expectation(description: "Raise remove flag")
        cluster.timeToRemove = 0.0
        var UUIDs: [UUID] = []
        for _ in 0...2 {
            UUIDs.append(UUID())
        }
        let pages = [
            Page(id: UUIDs[0], parentId: nil, title: nil, cleanedContent: "Roger Federer is the best tennis player to ever play the game, but Rafael Nadal is best on clay"),
            Page(id: UUIDs[1], parentId: UUIDs[0], title: nil, cleanedContent: "Tennis is a very fun game"),
            Page(id: UUIDs[2], parentId: UUIDs[0], title: nil, cleanedContent: "Pete Sampras and Roger Federer played 4 exhibition matches in 2008")
            ]
        for page in pages.enumerated() {
            cluster.add(page: page.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    if error as! Cluster.AdditionError != .skippingToNextAddition {
                        XCTFail(error.localizedDescription)
                    }
                case .success(let result):
                    if page.offset == pages.count - 1 {
                        expect(result.flag) == .sendRanking
                    }
                }
                if page.offset == pages.count - 1 {
                    expectation.fulfill()
                }
            })
        }
        wait(for: [expectation], timeout: 10)
    }

    /// Test that when a ranking is sent along with  a request to 'add', the 3 least ranked pages are removed.
    /// Also test different aspcets of beToghether and beApart matrices
    func testPageRemoval() throws {
        let cluster = Cluster()
        cluster.noteContentThreshold = 3
        // Here we don't want to test that notes with little content are not added
        let expectation = self.expectation(description: "Add page expectation")
        let beTogetherExpectation = self.expectation(description: "Be together expectation")
        var UUIDs: [UUID] = []
        for _ in 0...6 {
            UUIDs.append(UUID())
        }
        let pages = [
            Page(id: UUIDs[0], parentId: nil, title: "man", cleanedContent: "A man is eating food."),
            Page(id: UUIDs[1], parentId: UUIDs[0], title: "girl", cleanedContent: "The girl is carrying a baby."),
            Page(id: UUIDs[2], parentId: UUIDs[0], title: "man", cleanedContent: "A man is eating food."),
            Page(id: UUIDs[3], parentId: UUIDs[0], title: "girl", cleanedContent: "The girl is carrying a baby."),
            Page(id: UUIDs[4], parentId: UUIDs[0], title: "girl", cleanedContent: "The girl is carrying a baby."),
            Page(id: UUIDs[5], parentId: UUIDs[0], title: "man", cleanedContent: "A man is eating food."),
            Page(id: UUIDs[6], parentId: UUIDs[0], title: "fille", cleanedContent: "La fille est en train de porter un bébé.")
            ]
        for page in pages.enumerated() {
            var ranking: [UUID]?
            if page.offset == pages.count - 1 {
                ranking = [UUIDs[1], UUIDs[4], UUIDs[2], UUIDs[3], UUIDs[5], UUIDs[0]]
            }
            cluster.add(page: page.element, ranking: ranking, completion: { result in
                switch result {
                case .failure(let error):
                    if error as! Cluster.AdditionError != .skippingToNextAddition {
                        XCTFail(error.localizedDescription)
                    }
                case .success(let result):
                    _ = result
                }
                if page.offset == pages.count - 1 {
                    expectation.fulfill()
                }
            })
            if page.offset == 4 {
                let myUUID = UUID()
                let myNote = ClusteringNote(id: myUUID, title: "Roger Federer", content: ["Roger Federer is the best Tennis player in history"])
                cluster.add(note: myNote, ranking: nil, completion: { result in
                    switch result {
                    case .failure(let error):
                        if error as! Cluster.AdditionError != .skippingToNextAddition {
                            XCTFail(error.localizedDescription)
                        }
                    case .success(let result):
                        _ = result
                    }
                })
            }
        }
        wait(for: [expectation], timeout: 10)
        expect(cluster.adjacencyMatrix.rows) == 5 // 4 pages and one note
        expect(cluster.pages.count) == 4
        expect(cluster.notes.count) == 1
        expect(cluster.beTogetherMatrix.matrix.flat) == [Double](repeating: 0.0, count: 25)
        expect(cluster.beApartMatrix.matrix) == ones(5, 5) - diag([1.0, 1.0, 1.0, 1.0, 1.0])

        var pageBeTogether = Page(id: UUIDs[0])
        pageBeTogether.beWith = [UUIDs[3]]
        pageBeTogether.beApart = [UUIDs[5]]
        cluster.add(page: pageBeTogether, ranking: nil, completion: { result in
            switch result {
            case .failure(let error):
                if error as! Cluster.AdditionError != .skippingToNextAddition {
                    XCTFail(error.localizedDescription)
                }
            case .success(let result):
                _ = result
            }
            beTogetherExpectation.fulfill()
        })
        wait(for: [beTogetherExpectation], timeout: 10)
        var beWithMatrixCheck = [Double](repeating: 0.0, count: 25)
        beWithMatrixCheck[7] = 1.0
        beWithMatrixCheck[11] = 1.0
        var beApartMatrixCheck = (ones(5, 5) - diag([1.0, 1.0, 1.0, 1.0, 1.0])).flat
        beApartMatrixCheck[8] = 0.0
        beApartMatrixCheck[16] = 0.0
        expect(cluster.beTogetherMatrix.matrix.flat) == beWithMatrixCheck
        expect(cluster.beApartMatrix.matrix.flat) == beApartMatrixCheck
    }

    /// A page that was removed from the matrices is visited again by the user. Test that it is readded correctly and removed from attachedPages
    func testRevisitPageAfterRemoval() throws {
        let cluster = Cluster()
        let firstExpectation = self.expectation(description: "Add page expectation")
        let secondExpectation = self.expectation(description: "Add page expectation")
        var UUIDs: [UUID] = []
        for _ in 0...4 {
            UUIDs.append(UUID())
        }
        let firstPages = [
            Page(id: UUIDs[0], parentId: nil, title: "Page 1", cleanedContent: "A man is eating food."),
            Page(id: UUIDs[1], parentId: UUIDs[0], title: "Page 2", cleanedContent: "The girl is carrying a baby."),
            Page(id: UUIDs[2], parentId: UUIDs[0], title: "Page 3", cleanedContent: "A man is eating food.")
            ]
        let secondPages = [
            Page(id: UUIDs[3], parentId: UUIDs[0], title: "Page 4", cleanedContent: "The girl is carrying a baby."),
            Page(id: UUIDs[4], parentId: UUIDs[0], title: "Page 5", cleanedContent: "The girl is carrying a baby.")
            ]
        for page in (firstPages + secondPages).enumerated() {
            cluster.add(page: page.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    if error as! Cluster.AdditionError != .skippingToNextAddition {
                        XCTFail(error.localizedDescription)
                    }
                case .success(let result):
                    _ = result.0
                }
                if page.offset == (firstPages + secondPages).count - 1 {
                    firstExpectation.fulfill()
                }
            })
        }
        wait(for: [firstExpectation], timeout: 10)
        try cluster.remove(ranking: [UUIDs[3], UUIDs[4], UUID(), UUID(), UUID()], activeSources: [UUIDs[4]])
        expect(cluster.pages.count) == 4 // pages 3 was removed but page 4 was not
        expect(cluster.pages[3].id) == UUIDs[4]
        for page in secondPages.enumerated() {
            cluster.add(page: page.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    if error as! Cluster.AdditionError != .skippingToNextAddition {
                        XCTFail(error.localizedDescription)
                    }
                case .success(let result):
                    _ = result.0
                }
                if page.offset == secondPages.count - 1 {
                    secondExpectation.fulfill()
                }
            })
        }
        wait(for: [secondExpectation], timeout: 10)
        expect(cluster.pages.count) == 5
        expect(cluster.pages[4].id) == UUIDs[3]
        expect(cluster.beTogetherMatrix.matrix.flat) == [Double](repeating: 0.0, count: 25)
        expect(cluster.beApartMatrix.matrix) == ones(5, 5) - diag([1.0, 1.0, 1.0, 1.0, 1.0])
    }

    /// When removing a page from the matrix, chage that if the most similar data point to that page is a note, that does not create a problem
    func testRemovingPageWithSimilarNote() throws {
        let cluster = Cluster()
        cluster.noteContentThreshold = 3
        // Here we don't want to test that notes with little content are not added
        let expectation = self.expectation(description: "Add note expectation")
        var UUIDs: [UUID] = []
        for i in 0...5 {
            UUIDs.append(UUID())
            let myPage = Page(id: UUIDs[i], parentId: nil, title: nil, cleanedContent: "Here's some text for you")
            // The pages themselves don't matter as we will later force the similarity matrix
            cluster.add(page: myPage, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    if error as! Cluster.AdditionError != .skippingToNextAddition {
                        XCTFail(error.localizedDescription)
                    }
                case .success(let result):
                    _ = result.0
                }
            })
        }
        for i in 0...2 {
            let myNote = ClusteringNote(id: UUID(), title: "My note", content: ["This is my note"])
            cluster.add(note: myNote, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    if error as! Cluster.AdditionError != .notEnoughTextInNote  && error as! Cluster.AdditionError != .skippingToNextAddition {
                        XCTFail(error.localizedDescription)
                    }
                case .success(let result):
                    _ = result.0
                }
                if i == 2 {
                    expectation.fulfill()
                }
            })
        }
        wait(for: [expectation], timeout: 10)
        
        cluster.adjacencyMatrix = Matrix([[0, 0, 0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                                   [0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                   [0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                                   [0.9, 0.1, 0.3, 0, 0.5, 0.5, 0.5, 0.5, 0.5],
                                   [0.8, 0.2, 0.3, 0.5, 0, 0.7, 0.2, 0.1, 0.1],
                                   [0.7, 0.3, 0.3, 0.5, 0.7, 0, 0.6, 0.2, 0.1],
                                   [0.6, 0.4, 0.3, 0.5, 0.2, 0.6, 0, 0.9, 0.3],
                                   [0.5, 0.5, 0.3, 0.5, 0.1, 0.2, 0.9, 0, 0.4],
                                   [0.4, 0.6, 0.3, 0.5, 0.1, 0.1, 0.3, 0.4, 0]])
        try cluster.remove(ranking: [UUIDs[0]])
        expect(cluster.pages[0].id) == UUIDs[1]
    }

    /// Trying to add a note with little content should throw an expected error and not add the note
    func testNoteWithLittleContentIsNotAdded() throws {
        let cluster = Cluster()
        let firstShortNote = ClusteringNote(id: UUID(), title: "First short note", content: ["This is a short note"])
        let longNote = ClusteringNote(id: UUID(), title: "Roger Federer", content: ["Roger Federer (German: [ˈrɔdʒər ˈfeːdərər]; born 8 August 1981) is a Swiss professional tennis player. He is ranked No. 9 in the world by the Association of Tennis Professionals (ATP). He has won 20 Grand Slam men's singles titles, an all-time record shared with Rafael Nadal and Novak Djokovic. Federer has been world No. 1 in the ATP rankings a total of 310 weeks – including a record 237 consecutive weeks – and has finished as the year-end No. 1 five times. Federer has won 103 ATP singles titles, the second most of all-time behind Jimmy Connors, including a record six ATP Finals. Federer has played in an era where he dominated men's tennis together with Rafael Nadal and Novak Djokovic, who have been collectively referred to as the Big Three and are widely considered three of the greatest tennis players of all-time.[c] A Wimbledon junior champion in 1998, Federer won his first Grand Slam singles title at Wimbledon in 2003 at age 21. In 2004, he won three out of the four major singles titles and the ATP Finals,[d] a feat he repeated in 2006 and 2007. From 2005 to 2010, Federer made 18 out of 19 major singles finals. During this span, he won his fifth consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-ups to Nadal, his main rival up until 2010. At age 27, he also surpassed Pete Sampras's then-record of 14 Grand Slam men's singles titles at Wimbledon in 2009."])
        let secondShortNote = ClusteringNote(id: UUID(), title: "Second short note", content: ["This is a short note"])
        let myNotes = [firstShortNote, longNote, secondShortNote]
        let expectation = self.expectation(description: "Add note expectation")
        for aNote in myNotes.enumerated() {
            cluster.add(note: aNote.element, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    if error as! Cluster.AdditionError != .notEnoughTextInNote && error as! Cluster.AdditionError != .skippingToNextAddition {
                            XCTFail(error.localizedDescription)
                    }
                case .success(let result):
                    _ = result.0
                }
            })
        }
        let myPage = Page(id: UUID(), parentId: nil, title: "Page 1", cleanedContent: "A man is eating food.")
        cluster.add(page: myPage, ranking: nil, completion: { result in
            switch result {
            case .failure(let error):
                XCTFail(error.localizedDescription)
            case .success(let result):
                _ = result.0
                expectation.fulfill()
            }
        })
        wait(for: [expectation], timeout: 10)
        expect(cluster.notes.count) == 1
        expect(cluster.notes[0].id) == longNote.id
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
        cluster.entitiesMatrix.matrix = Matrix([[0, 0, 0, 0.4, 0.3, 0.2],
                                                [0, 0, 0, 0.5, 0.5, 0.5],
                                                [0, 0, 0, 0.1, 0.2, 0.3],
                                                [0.4, 0.5, 0.1, 0, 0.5, 0.2],
                                                [0.3, 0.5, 0.2, 0.5, 0, 0.3],
                                                [0.2, 0.5, 0.3, 0.2, 0.3, 0]])
        cluster.notes = [ClusteringNote(id: UUID(), title: "First note", content: ["note"]),
                         ClusteringNote(id: UUID(), title: "Second note", content: ["note"]),
                         ClusteringNote(id: UUID(), title: "Third note", content: ["note"])]
        cluster.pages = [Page(id: UUID(), parentId: nil, title: "First page", cleanedContent: "page"),
                         Page(id: UUID(), parentId: nil, title: "Second page", cleanedContent: "page"),
                         Page(id: UUID(), parentId: nil, title: "Third page", cleanedContent: "page")]
        let activeSources = [cluster.pages[0].id]
        let noteGroups = [[cluster.notes[0].id], [cluster.notes[1].id], [cluster.notes[2].id], [], []]
        let pageGroups = [[], [cluster.pages[2].id], [], [cluster.pages[0].id, cluster.pages[1].id]]
        
        let mySimilarities = cluster.createSimilarities(pageGroups: pageGroups, noteGroups: noteGroups, activeSources: activeSources)
        expect(mySimilarities[cluster.notes[0].id]) == [:]
        expect(mySimilarities[cluster.notes[1].id]) == [cluster.pages[2].id: 0.5625]
        expect(mySimilarities[cluster.notes[2].id]) == [:]
        expect(mySimilarities[cluster.pages[0].id]) == [cluster.pages[1].id: 0.5625]
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
    
    func testAddingDuringClustering() throws {
        let cluster = Cluster()
        cluster.isClustering = true
        var UUIDs: [UUID] = []
        for _ in 0...1 {
            UUIDs.append(UUID())
        }
        let pages = [
            Page(id: UUIDs[0], parentId: nil, title: "Page 1", cleanedContent: "A man is eating food."),
            Page(id: UUIDs[1], parentId: UUIDs[0], title: "Page 2", cleanedContent: "The girl is carrying a baby."),
            ]
        let firstPageExpectation = self.expectation(description: "Add page when isClustering is true")
        let secondPageExpectation = self.expectation(description: "Add page after isClustering is false again")
        cluster.add(page: pages[0], ranking: nil, completion: { result in
            switch result {
            case .failure(let error):
                if error as! Cluster.AdditionError == .abortingAdditionDuringClustering {
                    firstPageExpectation.fulfill()
                } else {
                    XCTFail("Expected error is aborting addition during clustering")
                }
            case .success:
                XCTFail("Page added during clustering")
            }
        })
        wait(for: [firstPageExpectation], timeout: 10)
        cluster.isClustering = false
        cluster.add(page: pages[1], ranking: nil, completion: { result in
            switch result {
            case .failure(let error):
                XCTFail(error.localizedDescription)
            case .success(let result):
                expect(result.pageGroups.flatMap{ $0 }.count) == 2
                secondPageExpectation.fulfill()
            }
        })
        wait(for: [secondPageExpectation], timeout: 10)
    }
    
    func testRemoveNote() throws {
        let cluster = Cluster()
        cluster.noteContentThreshold = 3
        // Here we don't want to test that notes with little content are not added
        let expectation = self.expectation(description: "Add note expectation")
        var pageUUIDs: [UUID] = []
        var noteUUIDs: [UUID] = []
        for i in 0...2 {
            pageUUIDs.append(UUID())
            let myPage = Page(id: pageUUIDs[i], parentId: nil, title: nil, cleanedContent: "Here's some text for you")
            // The pages themselves don't matter as we will later force the similarity matrix
            cluster.add(page: myPage, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    if error as! Cluster.AdditionError != .skippingToNextAddition {
                        XCTFail(error.localizedDescription)
                    }
                case .success(let result):
                    _ = result.0
                }
            })
        }
        for i in 0...2 {
            noteUUIDs.append(UUID())
            let myNote = ClusteringNote(id: noteUUIDs[i], title: "My note", content: ["This is my note"])
            cluster.add(note: myNote, ranking: nil, completion: { result in
                switch result {
                case .failure(let error):
                    if error as! Cluster.AdditionError != .notEnoughTextInNote  && error as! Cluster.AdditionError != .skippingToNextAddition {
                        XCTFail(error.localizedDescription)
                    }
                case .success(let result):
                    _ = result.0
                }
                if i == 2 {
                    expectation.fulfill()
                }
            })
            if i == 1 {
                cluster.removeNote(noteId: noteUUIDs[0])
            }
        }
        wait(for: [expectation], timeout: 10)
        expect(cluster.notes.count) == 2
        expect(cluster.notes[0].id) == noteUUIDs[1]
        expect(cluster.notes[1].id) == noteUUIDs[2]
    }
    
    func testTitlePreprocessing() throws {
        let cluster = Cluster()
        let expectation = self.expectation(description: "Add page expectation")
        let myPage = Page(id: UUID(), parentId: nil, url: URL(string: "http://www.cnn.com")!, title: "Roger Federer is the best tennis player ever | CNN", originalContent: [], cleanedContent: nil)
        cluster.add(page: myPage, ranking: nil, completion: { result in
            switch result {
            case .failure(let error):
                XCTFail(error.localizedDescription)
            case .success(let result):
                _ = result.0
            }
            expectation.fulfill()
        })
        wait(for: [expectation], timeout: 10)
        expect(cluster.pages[0].title) == "Roger Federer is the best tennis player ever | CNN"
        expect(cluster.pages[0].entitiesInTitle?.entities["PersonalName"]?[0]) == "roger federer"
    }

    func testGetExportInformationForIdPage() throws {
        let cluster = Cluster()
        let pageId = UUID()
        let expectation = self.expectation(description: "Add page expectation")
        let myPage = Page(id: pageId, parentId: nil, title: "Roger Federer", cleanedContent: "He was born on 8 August 1981 in Basel.")
        cluster.add(page: myPage, ranking: nil, completion: { result in
            switch result {
            case .failure(let error):
                XCTFail(error.localizedDescription)
            case .success(let result):
                _ = result.0
            }
            expectation.fulfill()
        })
        wait(for: [expectation], timeout: 10)
        //cluster.pages[0].language = NLLanguage.english // Normally cleaned content is only accepted in the method call when PnS is used. Here, in order to test getInformationForId we force language detection
        let pageInformation = cluster.getExportInformationForId(id: pageId)
        let expectedInformation = InformationForId(title: "Roger Federer", cleanedContent: "He was born on 8 August 1981 in Basel.", entitiesInText: EntitiesInText(entities: ["PersonalName": [String](), "PlaceName": ["basel"], "OrganizationName": [String]()]), entitiesInTitle: EntitiesInText(entities: ["PersonalName": ["roger federer"], "PlaceName": [String](), "OrganizationName": [String]()])/*, language: NLLanguage.english*/)
        expect(pageInformation) == expectedInformation
        let emptyInformation = cluster.getExportInformationForId(id: UUID())
        expect(emptyInformation) == InformationForId()
    }

    func testGetInformationForIdNote() throws {
        let cluster = Cluster()
        let noteId = UUID()
        let expectation = self.expectation(description: "Add note expectation")
        let myNote = ClusteringNote(id: noteId, title: "Roger Federer", content: ["Federer has played in an era where he dominated men's tennis along with Rafael Nadal and Novak Djokovic. Referred to as the Big Three, they are considered by some to be the three greatest tennis players of all time.[c] A Wimbledon junior champion in 1998, Federer won his first major singles title at Wimbledon in 2003 at age 21. In 2004, he won three of the four major singles titles and the ATP Finals,[d] a feat he repeated in 2006 and 2007. From 2005 to 2010, he made 18 out of 19 major singles finals. During this span, he won five consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-up finishes to Nadal, his main rival until 2010. At age 27, he surpassed Pete Sampras's record of 14 major men's singles titles at Wimbledon in 2009."])
        cluster.add(note: myNote, ranking: nil, completion: { result in
            switch result {
            case .failure(let error):
                XCTFail(error.localizedDescription)
            case .success(let result):
                _ = result.0
            }
            expectation.fulfill()
        })
        wait(for: [expectation], timeout: 10)
        let noteInformation = cluster.getExportInformationForId(id: noteId)
        let expectedInformation = InformationForId(title: "Roger Federer", cleanedContent: "Federer has played in an era where he dominated men\'s tennis along with Rafael Nadal and Novak Djokovic. Referred to as the Big Three, they are considered by some to be the three greatest tennis players of all time.[c] A Wimbledon junior champion in 1998, Federer won his first major singles title at Wimbledon in 2003 at age 21. In 2004, he won three of the four major singles titles and the ATP Finals,[d] a feat he repeated in 2006 and 2007. From 2005 to 2010, he made 18 out of 19 major singles finals. During this span, he won five consecutive titles at both Wimbledon and the US Open. He completed the career Grand Slam at the 2009 French Open after three previous runner-up finishes to Nadal, his main rival until 2010. At age 27, he surpassed Pete Sampras\'s record of 14 major men\'s singles titles at Wimbledon in 2009.", entitiesInText: EntitiesInText(entities: ["PlaceName": ["wimbledon"], "PersonalName": ["federer", "rafael nadal", "novak djokovic", "nadal", "pete sampras"], "OrganizationName": ["atp finals"]]), entitiesInTitle: EntitiesInText(entities: ["PersonalName": ["roger federer"], "PlaceName": [String](), "OrganizationName": [String]()])/*, language: NLLanguage.english*/)
        expect(noteInformation) == expectedInformation
        let emptyInformation = cluster.getExportInformationForId(id: UUID())
        expect(emptyInformation) == InformationForId()
    }

    func testPreparePersonName() throws {
        let cluster = Cluster()
        let foundNames = ["Roger Federer", "Joe Biden", "joe", "Federer", "Nadal"]
        let preparedNames = cluster.preparePersonName(namesFound: foundNames)
        expect(Set(preparedNames)) == Set([["roger", "federer"], ["joe", "biden"], ["nadal"]])
    }

    func testComparePersonNames () throws {
        let cluster = Cluster()
        let preparedNames1 = [["roger", "federer"], ["joe", "biden"], ["nadal"]]
        let preparedNames2 = [["roger", "federer"], ["rafael", "nadal"], ["novak", "djokovic"]]
        let score = cluster.comparePersonNames(names1: preparedNames1, names2: preparedNames2)
        expect(score) == 1.0
    }

    func testRemoveDomainFromEntities() throws {
        var entitiesInText = EntitiesInText()
        entitiesInText.entities["PersonalName"] = ["roger federer", "joe bien", "roger federer"]
        entitiesInText.entities["OrganizationName"] = ["amazon", "wimbledon"]
        entitiesInText.entities["PlaceName"] = ["wimbledon"]
        let domainTokens = ["www", "amazon", "com"]
        let cluster = Cluster()
        let newEntities = cluster.removeDomainFromEntities(entitiesInText: entitiesInText, domainTokens: domainTokens)
        expect(newEntities.entities["PersonalName"]) == ["roger federer", "joe bien", "roger federer"]
        expect(newEntities.entities["OrganizationName"]) == ["wimbledon"]
        expect(newEntities.entities["PlaceName"]) == ["wimbledon"]
    }

    func testCheckShortNote() throws {
        let cluster = Cluster()
        let expectation = self.expectation(description: "Add short note expectation")
        let myNote = ClusteringNote(id: UUID(), title: "Roger Federer", content: ["Roger Federer is the best Tennis player in history"])
        var myFailure: String?
        cluster.add(note: myNote, ranking: nil, completion: { result in
            switch result {
            case .failure(let error):
                if error as! Cluster.AdditionError == .notEnoughTextInNote {
                    myFailure = error.localizedDescription
                }
                expectation.fulfill()
            case .success(let result):
                _ = result
                expectation.fulfill()
            }
        })
        wait(for: [expectation], timeout: 10)
        XCTAssertTrue((myFailure ?? "").hasSuffix("(Clustering.Cluster.AdditionError erreur 2.)"))
    }

    func testbeWithAndBeApart() throws {

    }
    // swiftlint:disable:next file_length
}
