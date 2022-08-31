//
//  MathUtilsTests.swift
//  
//
//  Created by Julien Plu on 19/07/2022.
//

import XCTest
@testable import Clustering
import Nimble


class MathsUtilsTests: XCTestCase {
    func testCosineSimilarity() throws {
        let vec1 = [0.0, 1.5, 3.0, 4.5, 6.0]
        let vec2 = [2.0, 4.0, 6.0, 8.0, 10.0]
        let cossim = MathsUtils.cosineSimilarity(vector1: vec1, vector2: vec2)

        expect(cossim).to(beCloseTo(0.9847319278346619, within: 0.0001))
    }

    func testNormalize() throws {
        let vec = [0.41072642, 0.3150499 , 0.04462959, 0.87794215, 0.26352442,
                   0.427552  , 0.25870064, 0.52616206, 0.21725976, 0.30784259]
        let normalized_v = MathsUtils.normalize(vector: vec)
        
        expect(normalized_v).to(beCloseTo([0.30796373, 0.23622523, 0.03346338, 0.65828329, 0.19759129,
                                           0.32057959, 0.19397441, 0.39451767, 0.16290193, 0.23082117], within: 0.0001))
    }
}
