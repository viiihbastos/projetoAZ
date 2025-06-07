import unittest, re
from .test_spark_base import *
from databricksx12.edi import *

class TestEDI(PysparkBaseTest):

    x = EDI(open("sampledata/837/CC_837I_EDI.txt", "rb").read().decode("utf-8"))
    y = EDI(open("sampledata/837/CC_837P_EDI.txt", "rb").read().decode("utf-8"))
    z = EDI(open("sampledata/837/CHPW_Claimdata.txt", "rb").read().decode("utf-8"))

    def test_no_isa(self):
        x = EDI(open("sampledata/malformed_files/CC_837I_EDI.txt", "rb").read().decode("utf-8"))
        assert(x.to_json().get('EDI.control_number') == "")
        
    
    #
    # Test counting total number of segments 
    # 
    def test_segment_count(self):
        assert ( TestEDI.x.segment_count() == 66)
        assert ( TestEDI.y.segment_count() == 34)

    def test_get_segment_by_name(self):
        assert ( len(TestEDI.x.segments_by_name("ISA")) == 1)
        assert ( len(TestEDI.y.segments_by_name("ISA")) == 1)
        assert ( len(TestEDI.x.segments_by_name("NM1")) == 6)
        assert ( len(TestEDI.y.segments_by_name("NM1")) == 5)
        assert ( set([ isinstance(type(x), type(Segment)) for x in TestEDI.x.segments_by_name("NM1") ]) == {True} )

    def test_segments_by_position(self):
        assert(TestEDI.x.segments_by_position(0,1)[0].segment_name() == "ISA")
        assert(TestEDI.y.segments_by_position(0,1)[0].segment_name() == "ISA")
        assert(len(TestEDI.x.segments_by_position(0,1)) == 1)
        assert(len(TestEDI.x.segments_by_position(0,100)) == 66)
        assert(TestEDI.x.segments_by_position(15, 20)[0].segment_name() == "NM1")


    def test_functional_groups(self):
        assert(len(TestEDI.z.functional_segments()) == 1)
        assert(len(TestEDI.z.functional_segments()[0].data) == 173)
        assert(TestEDI.z.num_transactions() == 5)
        assert(TestEDI.z.num_functional_groups() == 1)
        assert(TestEDI.z.functional_segments()[0].data[0].element(0) == "GS" and TestEDI.z.functional_segments()[0].data[172].element(0) == "GE")
        assert(len([a.element(0) for a in TestEDI.z.functional_segments()[0].data if a.element(0) == "ST"]) == TestEDI.z.num_transactions())
            
if __name__ == '__main__':
    unittest.main()
