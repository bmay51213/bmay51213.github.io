---
layout: archive
permalink: /machine-learning/
title: "Machine Learning Projects"
author_profile: true
header:
  image: "/images/top-5-data-science-strategy-predictions-2020-1024x440.jpg"
---

{% include base_path %}
{% include group-by-array
collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts =
  group_items[forloop.index0] %}
  <h2 id="{{ tag  |  slugify}}"
  class = "archive_subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
