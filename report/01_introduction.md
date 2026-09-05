# 1. Introduction

## Business problem

An online toy shop needs relevant suggestions on each product page. The shelf
must balance alternatives similar to the viewed product with complementary
products frequently bought with it. Useful recommendations should therefore
combine direct basket evidence, positions in the purchase graph, and attributes
such as category, age, brand, or hero in one ranked list.

## Objective

The project builds a local, reproducible recommendation system that:

- discovers frequent product sets;
- represents SKU relationships as a normalized co-purchase graph;
- learns dense product embeddings from weighted Node2vec walks;
- models Greek/English product-name content with word/character TF-IDF;
- presents a combined one-SKU product-page recommendation shelf;
- predicts missing links with Adamic–Adar;
- supports metadata restrictions;
- evaluates recommendations on later, unseen orders; and
- produces qualitative examples for manual error analysis.

## Scope

The current system recommends products rather than users. The deployed
Streamlit interface begins from one selected product, while the offline
basket-completion experiment may use one or more observed products to measure
retrieval of a hidden order partner. It uses order-level baskets and static
catalog metadata; it does not use personal profiles, customer identifiers,
prices, margin, or item-level event sequences. All training and evaluation run
locally without an API key.
