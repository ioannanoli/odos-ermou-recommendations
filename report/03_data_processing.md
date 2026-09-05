# 3. Data Processing

## Loading and validation

`load_orders` validates the required order ID, order status, and SKU columns.
It reads Excel through pandas/openpyxl, optionally applies a status allow-list,
and removes rows that cannot identify an order or product.

## Text normalization

Object columns are HTML-unescaped, normalized with Unicode NFKC, stripped, and
collapsed to single whitespace. Common textual missing markers such as `nan`,
`none`, `null`, and `n/a` become actual missing values.

## Product catalog

`create_product_catalog` produces one row per SKU. When repeated order lines
contain conflicting metadata, the most frequent non-null value is retained.
An attribute with no observed value becomes `Unknown`.

## Basket construction

Lines are grouped by order ID and duplicate SKUs within one order are collapsed.
This means graph and itemset counts measure the number of orders containing a
product or pair, not purchased quantity.

## Edge normalization

For products `i` and `j`, the graph stores raw pair count `c(i,j)` and cosine-
style normalized weight:

```text
w(i,j) = c(i,j) / sqrt(c(i) * c(j))
```

This reduces the tendency of globally popular products to dominate purely
because they appear in many orders.
