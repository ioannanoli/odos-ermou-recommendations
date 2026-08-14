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

## Privacy

The export also contains billing and shipping location fields. The raw workbook
is never committed, copied into reports, or written to output CSVs. Versioned
outputs contain product-level or aggregate information only.
