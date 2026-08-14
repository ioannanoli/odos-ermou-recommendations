# 1. Introduction

## Business problem

An online toy shop needs relevant product suggestions for customers viewing an
item or assembling a cart. Useful recommendations should reflect products that
are purchased together, products occupying similar positions in the purchase
graph, plausible relationships not yet observed directly, and constraints such
as age suitability or product category.

## Objective

The project builds a local, reproducible recommendation system that:

- discovers frequent product sets;
- represents SKU relationships as a normalized co-purchase graph;
- learns dense product embeddings from weighted Node2vec walks;
- recommends products for one SKU or a multi-product cart;
- predicts missing links with Adamic–Adar;
- supports metadata restrictions;
- evaluates recommendations on later, unseen orders; and
- produces qualitative examples for manual error analysis.

## Scope

The current system recommends products rather than users. It uses order-level
baskets and static catalog metadata; it does not use personal profiles,
customer identifiers, prices, inventory, margin, or item-level event sequences.
All training and evaluation run locally without an API key.
