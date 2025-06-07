from .test_spark_base import *
from databricksx12.edi import *

class TestPyspark(PysparkBaseTest):

    def test_transaction_count(self):
        df = self.spark.read.text("sampledata/837/*txt", wholetext=True)
        data = (df.rdd
                .map(lambda x: x.asDict().get("value"))
                .map(lambda x: EDI(x))
                .map(lambda x: {"transaction_count": x.num_transactions()})
                ).toDF()
        assert ( data.count() == 5) #5 rows
        assert ( data.select(data.transaction_count).groupBy().sum().collect()[0]["sum(transaction_count)"] == 9) #8 ST/SE transactions


if __name__ == '__main__':
    unittest.main()
