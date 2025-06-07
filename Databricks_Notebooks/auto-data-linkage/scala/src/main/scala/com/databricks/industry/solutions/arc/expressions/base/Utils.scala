package com.databricks.industry.solutions.arc.expressions.base

import org.apache.spark.sql.catalyst.util.{ArrayBasedMapBuilder, ArrayBasedMapData, ArrayData}
import org.apache.spark.sql.types.{DoubleType, LongType, StringType}
import org.apache.spark.unsafe.types.UTF8String

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

object Utils {

    /**
     * Builds a spark map from a scala Map[String, Long].
     * @param metaData
     *   The metadata to be used.
     * @return
     *   Serialized map.
     */
    def buildMapLong(metaData: Map[String, Long]): ArrayBasedMapData = {
        val keys = ArrayData.toArrayData(metaData.keys.toArray[String].map(UTF8String.fromString))
        val values = ArrayData.toArrayData(metaData.values.toArray[Long])
        val mapBuilder = new ArrayBasedMapBuilder(StringType, LongType)
        mapBuilder.putAll(keys, values)
        mapBuilder.build()
    }

    /**
     * Builds a spark map from a scala Map[String, Double].
     * @param metaData
     *   The metadata to be used.
     * @return
     *   Serialized map.
     */
    def buildMapDouble(metaData: Map[String, Double]): ArrayBasedMapData = {
        val keys = ArrayData.toArrayData(metaData.keys.toArray[String].map(UTF8String.fromString))
        val values = ArrayData.toArrayData(metaData.values.toArray[Double])
        val mapBuilder = new ArrayBasedMapBuilder(StringType, DoubleType)
        mapBuilder.putAll(keys, values)
        mapBuilder.build()
    }

    def serialize(input: Any): Array[Byte] = {
        val byteStream = new ByteArrayOutputStream()
        val writer = new ObjectOutputStream(byteStream)
        writer.writeObject(input)
        writer.flush()
        byteStream.toByteArray
    }

    def deserialize[T](bytes: Array[Byte]): T = {
        val byteStream = new ByteArrayInputStream(bytes)
        val reader = new ObjectInputStream(byteStream)
        reader.readObject().asInstanceOf[T]
    }

}
