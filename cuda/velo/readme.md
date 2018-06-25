VELO
=====

Velo Tracking

Masked Clustering
-----------------------

Format needed by one RawEvent for Daniel's clustering code:

```
-----------------------------------------------------------------------------
name                |  type    |  size                 | array_size
=============================================================================
number_of_rawbanks  | uint32_t | 1
-----------------------------------------------------------------------------
raw_bank_offset     | uint32_t | number_of_rawbanks
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
sensor_index        | uint32_t | 1                     |
------------------------------------------------------------------------------
sp_count            | uint32_t | 1                     | number_of_rawbanks
------------------------------------------------------------------------------
sp_word             | uint32_t | sp_count              |
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
```

raw_bank_offset: array containing the offsets to each raw bank, this array is 
currently filled when running Brunel to create this output; it is needed to process
several raw banks in parallel

sp = super pixel

sp_word: contains super pixel address and 8 bit hit pattern
