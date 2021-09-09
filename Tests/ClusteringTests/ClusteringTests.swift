import Nimble
import XCTest
import LASwift
import NaturalLanguage
@testable import Clustering

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

    

}
