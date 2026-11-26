#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xtier_kv_store xtier_kv_store_t;

typedef struct xtier_kv_key {
    uint64_t lo;
    uint64_t hi;
} xtier_kv_key_t;

typedef struct xtier_kv_iov {
    void* base;
    size_t len;
} xtier_kv_iov_t;

typedef struct xtier_kv_item {
    uint16_t pool_id;
    uint16_t reserved;
    uint32_t iov_count;
    xtier_kv_key_t key;
    const xtier_kv_iov_t* iovs;
} xtier_kv_item_t;

typedef struct xtier_kv_config {
    uint64_t bytes_per_drive;   // e.g. 50 GiB
    uint64_t region_bytes;      // default 256 MiB (must divide into >= 4 regions)
    uint32_t write_qd;          // default 4 per drive
    uint32_t read_qd;           // default 64 per drive (0 = ring_depth - 4)
    uint32_t bounce_bytes;      // default 32 MiB per drive
    uint32_t relocate_pct;      // default 50 (max % of region_bytes relocated per recycle)
    uint32_t ring_depth;        // default 128
    uint32_t flags;             // bit 0: pin_numa (default 1)
} xtier_kv_config_t;

typedef struct xtier_kv_stats {
    uint64_t live_entries;
    uint64_t live_bytes;
    uint64_t capacity_bytes;
    uint64_t puts_ok;
    uint64_t puts_skipped_dup;
    uint64_t puts_failed;
    uint64_t gets_hit;
    uint64_t gets_miss;
    uint64_t gets_failed;
    uint64_t bytes_written;
    uint64_t bytes_read;
    uint64_t regions_recycled;
    uint64_t hot_relocated;
    uint64_t hot_dropped_cap;
    uint64_t direct_ios;
    uint64_t bounce_ios;
    uint64_t evict_wait_count;
} xtier_kv_stats_t;

xtier_kv_store_t* xtier_kv_open(const char* const* paths, int num_drives, const xtier_kv_config_t* cfg);
void xtier_kv_close(xtier_kv_store_t* store);
void xtier_kv_clear(xtier_kv_store_t* store);
void xtier_kv_stats_get(xtier_kv_store_t* store, xtier_kv_stats_t* out);
const char* xtier_kv_last_error(void);

void xtier_kv_exists(xtier_kv_store_t* store, uint16_t pool_id, const xtier_kv_key_t* keys, int count,
                     uint8_t* out_bitmap);
int xtier_kv_put(xtier_kv_store_t* store, const xtier_kv_item_t* items, int count, uint8_t* out_ok);
int xtier_kv_get(xtier_kv_store_t* store, const xtier_kv_item_t* items, int count, uint8_t* out_ok);
void xtier_kv_set_test_read_delay_us(xtier_kv_store_t* store, uint32_t delay_us);

#ifdef __cplusplus
}
#endif
