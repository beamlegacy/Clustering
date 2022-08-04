//
//  MathsUtils.swift
//  
//
//  Created by Julien Plu on 19/07/2022.
//

import Accelerate


extension Double {
    /// Rounds the double to decimal places value
    func rounded(toPlaces places:Int) -> Double {
        let divisor = pow(10.0, Double(places))
        return (self * divisor).rounded() / divisor
    }
}


class MathsUtils {
    /// Compute the normalized vector of a given vector.
    ///
    /// - Parameters:
    ///   - vector: The vector to be normalized.
    /// - Returns: The normalized vector of the given vector.
    static func normalize(vector: [Double]) -> [Double] {
        let normValue = cblas_dnrm2(Int32(vector.count), vector, 1)
        //fputs("Swift: " + String(normValue) + "\n", stderr)
        if normValue > 0 {
            var normalizedVector = [Double]()
            
            for i in 0...vector.count - 1 {
                normalizedVector.append((vector[i] / normValue))
            }
            
            return normalizedVector
        }
        
        return vector
    }

    /// Compute the cosine similarity between two vectors
    ///
    /// - Parameters:
    ///   - vector1: a vector
    ///   - vector2: a vector
    /// - Returns: The cosine similarity between the two given vectors.
    static func cosineSimilarity(vector1: [Double], vector2: [Double]) -> Double {
        let vector1Norm = normalize(vector: vector1)
        let vector2Norm = normalize(vector: vector2)
        
        if vector1Norm == [Double](repeating: 0.0, count: vector1.count) || vector2Norm == [Double](repeating: 0.0, count: vector1.count) {
            return 0.0//.rounded(toPlaces: 4)
        }
        
        let vector1NormVector2NormDotProduct = cblas_ddot(Int32(vector1Norm.count), vector1Norm, 1, vector2Norm, 1)
        let vector1NormDotProduct = cblas_ddot(Int32(vector1Norm.count), vector1Norm, 1, vector1Norm, 1)
        let vector2NormDotProduct = cblas_ddot(Int32(vector2Norm.count), vector2Norm, 1, vector2Norm, 1)
        let sqrtVector1NormDotProduct = pow(vector1NormDotProduct, 0.5)
        let sqrtVector2NormDotProduct = pow(vector2NormDotProduct, 0.5)
        let similarity = vector1NormVector2NormDotProduct / (sqrtVector1NormDotProduct * sqrtVector2NormDotProduct)
        //fputs("Swift: " + String(similarity.rounded(toPlaces: 4)) + "\n", stderr)
        return similarity//.rounded(toPlaces: 4)
    }
}
