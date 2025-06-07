from .test_spark_base import *
from .test_pyspark import *
from databricksx12.hls import *
from databricksx12 import *
import unittest, re
from functools import reduce
from operator import add


class TestClaims(PysparkBaseTest):


    def test_no_isa(self):
        edi = EDI(open("sampledata/malformed_files/CC_837I_EDI.txt", "rb").read().decode("utf-8"))
        hm = HealthcareManager()
        data = hm.from_edi(edi)[0]
        assert(edi.to_json().get('EDI.control_number') == "")
        assert([y.to_dict().get("claim_line_number") for y in data.sl_info] == ['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        assert( reduce(add, [float(y.to_dict().get("line_chrg_amt")) for y in data.sl_info]) == 17166.7)
        assert(len(hm.to_json(edi).get("FunctionalGroup")) == 1)
        

    def test_professional_service_lines(self):
        edi = EDI(open("sampledata/837/CC_837P_EDI.txt", "rb").read().decode("utf-8"))
        hm = HealthcareManager()
        data = hm.from_edi(edi)[0]
        assert(len(data.sl_info) == 2)
        assert([y.to_dict().get("claim_line_number") for y in data.sl_info] == ['1', '2'])
        assert([y.to_dict().get("place_of_service") for y in data.sl_info] == ['11', '11'])
        assert([y.to_dict().get("line_chrg_amt") for y in data.sl_info] == ['300', '300'])
        
    def test_institutional_service_lines(self):
        edi = EDI(open("sampledata/837/CC_837I_EDI.txt", "rb").read().decode("utf-8"))
        hm = HealthcareManager()
        data = hm.from_edi(edi)[0]
        assert([y.to_dict().get("claim_line_number") for y in data.sl_info] == ['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        assert([y.to_dict().get("revenue_cd") for y in data.sl_info] ==['0124', '0250', '0260', '0300', '0301', '0305', '0306', '0307', '0351'])
        assert( reduce(add, [float(y.to_dict().get("line_chrg_amt")) for y in data.sl_info]) == 17166.7)

if __name__ == '__main__':
    unittest.main()        
