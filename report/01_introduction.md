# 1. Introduction

## Business problem

An online toy shop needs relevant suggestions on each product page. Two customer
questions are treated separately: **Which alternatives are similar to this
product?** and **Which complementary products are frequently bought with it?**
Useful recommendations should reflect direct basket evidence, products occupying
similar positions in the purchase graph, and attributes such as category, age,
brand, or hero.

## Objective

The project builds a local, reproducible recommendation system that:

- discovers frequent product sets;
- represents SKU relationships as a normalized co-purchase graph;
- learns dense product embeddings from weighted Node2vec walks;
- presents one-SKU product-page recommendations in separate **Similar items**
  and **Frequently bought together** sections;
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
