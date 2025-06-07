package com.databricks.industry.solutions.arc.expressions

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.test.SharedSparkSession

class ARC_GenerateCombinationsTest extends QueryTest with SharedSparkSession with ARC_GenerateCombinationsBehaviors {

    test("ARC_GenerateCombinations expression") { testGenerateCombinations() }

    test("ARC_GeneratePartialCombinations expression") { testGeneratePartialCombinations() }

}
