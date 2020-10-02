import tables
import numpy as np
table_desc = tables.open_file('./data/test.desc.h5')
descs = table_desc.get_node('/phrases')[:]
idx_descs = table_desc.get_node('/indices')[:]
len, pos = idx_descs[1]['length'], idx_descs[1]['pos']
good_desc = descs[pos:pos + len]
print(len)
print(good_desc)