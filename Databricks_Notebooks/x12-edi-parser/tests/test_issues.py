from .test_spark_base import *
from .test_pyspark import *
from databricksx12.hls import *
from databricksx12 import *
import unittest, re
from os import listdir
from os.path import isfile, join

class TestIssues(PysparkBaseTest):

    #Pulling delimiters from ISA 
    def test_issue21(self):
        files = ['sampledata/837/' + f for f in listdir('sampledata/837') if isfile(join('sampledata/837', f))]
        assert(set(map(lambda f:
                 EDI.extract_delim(open(f, "rb").read().decode("utf-8")) == AnsiX12Delim
            , files)) == {True})

    #claim header level date/time and date qualifier codes
    def test_issue24(self):
        edi = EDI(open('sampledata/837/CC_837I_EDI.txt', "rb").read().decode("utf-8"))
        hm = HealthcareManager()
        data = hm.from_edi(edi)
        assert(len(data[0].to_json()['claim_header']['claim_dates']))
        assert([x['date_cd'] for x in data[0].to_json()['claim_header']['claim_dates']] == ['096', '434', '435'])
        assert([x['date'] for x in data[0].to_json()['claim_header']['claim_dates']] == ['0900', '20180628-20180702', '201806280800'])
        assert([x['date_format'] for x in data[0].to_json()['claim_header']['claim_dates']] == ['TM', 'RD8', 'DT'])

    
    def test_issue19(self):
        edi = EDI(open('sampledata/835/sample.txt', 'rb').read().decode("utf-8"))
        hm = HealthcareManager()
        data = hm.from_edi(edi)
        assert(len(data) == 3)
        assert(len(data[0].to_json()['provider_adjustments']) == 0)
        assert(len(data[1].to_json()['provider_adjustments']) == 0)
        assert(len(data[2].to_json()['provider_adjustments']) == 1)

        edi = EDI(open('sampledata/835/plb_sample.txt', 'rb').read().decode("utf-8"))
        data = hm.from_edi(edi)
        assert(len(data) == 1)
        assert(len(data[0].to_json()['provider_adjustments']) == 1)

    #capture all other dx codes
    def test_issue15(self):
        edi = EDI(open('sampledata/837/CC_837I_EDI.txt', "rb").read().decode("utf-8"))
        hm = HealthcareManager()
        data = hm.from_edi(edi)
        assert(data[0].to_json()['diagnosis']['other_dx_cds'] == 'F1319,F419,F17210,E876')


    def test_issue25_no_cas(self):
        edi = EDI(open('sampledata/835/sample_no_cas.txt', "rb").read().decode("utf-8"))
        hm = HealthcareManager()
        data = hm.from_edi(edi)
        assert(data[0].to_json()['claim']['service_adjustments'] == [])
        assert(len(data[1].to_json()['claim']['service_adjustments']) == 0)
        assert(len(data[2].to_json()['claim']['service_adjustments']) == 1)
        assert(len(data[3].to_json()['claim']['service_adjustments']) == 2)
        

if __name__ == '__main__':
    unittest.main()
