from enum import IntEnum

DataProcessTypes = IntEnum('DataProcessTypes', ('standardization', 'normalization', 'no_preprocess'))
SetType = IntEnum('SetType', ('train', 'valid', 'test'))
