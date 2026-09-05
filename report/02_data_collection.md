# 2. Data Collection

## Source

The source is a WooCommerce Excel order export named
`Orders-Export-2026-June-07-2054.xlsx`. The private workbook is intentionally
ignored by Git and must be supplied locally in `data/`.

## Observed coverage

After rows without an order ID or usable SKU are removed, the analytical data
contains:

- 14,178 order lines;
- 9,727 distinct orders;
- 6,257 distinct SKUs; and
- order dates from 15 April 2019 through 19 June 2026.

The export contains order ID, order date, order status, SKU, product name,
quantity and value fields, and Greek/English product metadata such as category,
brand, age, hero, and gender.

## Status policy

No status is excluded by default. Completed, cancelled, pending, refunded,
failed, pickup, and other exported statuses remain in the data, following the
defined project requirement. This policy should be revisited before production
because cancelled or failed baskets may describe intent rather than purchases.

## Untouched future export

A second WooCommerce export supplied as a UTF-8 CSV was reserved for the
strictly future evaluation. The private source file remains outside the
repository. After applying the frozen cutoff of 19 June 2026 at 14:28:25,
removing order IDs already present in model history, and retaining every order
status, it contains:

- 370 order lines from 287 new orders;
- dates from 19 June 2026 at 22:16:17 through 25 August 2026 at 13:20:29;
- 253 unique SKUs, of which 153 were known to frozen history and 100 were new;
- 45 multi-product orders; and
- 24 orders with at least two SKUs known to the frozen catalog, which form the
  eligible one-SKU evaluation queries.

Cancelled orders were deliberately retained. The output configuration records
the source file's SHA-256 fingerprint so the evaluation can be reproduced
without publishing customer-level raw data.

## Privacy

The export also contains billing and shipping location fields. The raw workbook
and future CSV are never committed, copied into reports, or written to output
CSVs. Versioned outputs exclude customer names, contact details, addresses, and
locations. Reproducibility tables retain only internal order IDs, SKUs, ranks,
and aggregate metrics.
