# Bing Maps Overlay Endpoint

## Overview

Bing Maps exposes a lightweight overlay endpoint at `/maps/overlaybfpr` that returns
business listings as HTML fragments. The same endpoint serves two modes:

1. **Discovery mode** (category query, count=10): Returns 10 venue cards with basic info (~147KB)
2. **Detail mode** (specific venue query, count=1): Returns full detail panel (~330KB) with
   reviews, hours, description, social media, photos

No browser or JavaScript execution required — pure HTTP GET requests.

## Endpoint

```
GET https://www.bing.com/maps/overlaybfpr?q={query}&cp={lat}~{lng}&count={count}
```

**Required headers:**
- `Referer: https://www.bing.com/maps`
- `Accept: text/html`

**Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `q` | Search query (URL-encoded) | `wedding+venues` or `Allan+House+Austin+TX` |
| `cp` | Center point as `lat~lng` | `30.2672~-97.7431` |
| `count` | Results per page | `10` (discovery) or `1` (detail) |
| `pos` | Pagination offset (1-indexed) | `11` for page 2 with count=10 |

## Two-Phase Strategy

### Phase 1: Discovery (count=10)

Query: `?q=wedding+venues&cp=30.2672~-97.7431&count=10`
Response: ~147KB, 10 venue cards

**14 fields per venue (verified 2026-02-10):**

| Field | Source | Coverage |
|-------|--------|----------|
| name | `data-entity` → `entity.title` | 100% |
| address | `data-entity` → `entity.address` | 100% |
| phone | `data-entity` → `entity.phone` | 100% |
| website | `data-entity` → `entity.website` | 100% |
| category | `data-entity` → `entity.primaryCategoryName` | 100% |
| style_category | `data-entity` → `entity.primaryStyleCategory` | 100% |
| ypid | `data-entity` → `entity.id` | 100% |
| latitude | `data-entity` → `geometry.y` | 100% |
| longitude | `data-entity` → `geometry.x` | 100% |
| image_url | `data-entity` → `entity.imageUrl` | 90% |
| rating | factrow text regex `(\d\.?\d?)/5.*?\((\d+)\)` | 70% |
| review_count | factrow text regex (same) | 70% |
| hours_status | factrow text regex `(Open\|Closed)` | 70% |
| hours_detail | factrow text after status | 70% |

**DOM structure:**
```
<li>
  <a class="listings-item" data-entity="{...JSON...}">
    <div class="b_factrow">Business Name</div>
    <div class="b_factrow">4.5/5(47)· Category</div>
    <div class="b_factrow">Address</div>
    <div class="b_factrow">Open· Closes 6 PM</div>
    <div class="b_factrow">(512) 123-4567</div>
  </a>
</li>
```

Factrows are CHILDREN of the `<a>` data-entity element, not siblings.

### Phase 2: Detail (count=1, specific venue)

Query: `?q=Chateau+Bellevue+Austin+TX&cp=30.2708~-97.7481&count=1`
Response: ~330KB, single venue detail panel

**21-22 fields per venue (verified 2026-02-10):**

Everything from Phase 1, plus:

| Field | Source | Notes |
|-------|--------|-------|
| rating | `revdata` JSON → `Provider.AverageRating` | Structured, reliable |
| review_count | `revdata` JSON → `Provider.ReviewCount` | Structured |
| review_source | `revdata` JSON → `Provider.Name` | "Yelp", "Tripadvisor" |
| review_source_url | `revdata` JSON → `Provider.Link.Url` | Full Yelp/TA URL |
| reviews[] | `revdata` JSON → `Reviews.Values[]` | 2-3 individual reviews |
| all_reviews_url | `revdata` JSON → `SeeAll.Url` | "See all on Yelp" link |
| description | `.ed_entity_desc` text | About text |
| hours | `.opHours` text (day-by-day regex) | Structured per-day hours |
| social_media | `.socialProfile` links | FB, X, IG, Pinterest URLs |
| address_structured | `data-facts` JSON → `addressFields` | street/city/state/zip |
| photos[] | `img[src*=bing.com/th]` with OLC. | Multiple thumbnail URLs |
| bing_category_id | `data-entitymetadata` JSON → `PCID` | Bing internal cat ID |

## Detail Data Sources

### 1. `revdata` JSON attribute (reviews)

The `.reviews_rct` element has a `revdata` attribute containing fully structured review data:

```json
{
  "Reviews": {
    "Values": [
      {
        "Text": "What a delight of a venue...",
        "Rating": {
          "ReviewRating": 10,
          "ProviderName": "Yelp",
          "ReviewerName": "Rob Z.",
          "ReviewTimeStamp": "Apr 5, 2025"
        },
        "FullReviewLink": {
          "Url": "https://www.yelp.com/biz/..."
        }
      }
    ]
  },
  "Provider": {
    "Name": "Yelp",
    "ReviewCount": 41,
    "AverageRating": 4.5,
    "Link": {"Url": "https://www.yelp.com/biz/..."}
  },
  "SeeAll": {
    "Text": "See all reviews on Yelp",
    "Url": "https://www.yelp.com/biz/..."
  }
}
```

**Notes:**
- `ReviewRating` is 0-10 scale (divide by 2 for 5-star)
- 2-3 reviews per response (most recent)
- `reqbaseurl` attribute on `.reviews_rct` has `/local/mprreviewsdata?...` endpoint for more reviews

### 2. `.opHours` element (structured hours)

```
Closed· Opens tomorrow 9:30 AM
Days of week  Open hours
Tuesday       9:30 AM - 4 PM
Wednesday     9:30 AM - 4 PM
Thursday      9:30 AM - 4 PM
Friday        9:30 AM - 3 PM
Saturday      Closed
Sunday        Closed
Monday        9:30 AM - 4 PM
```

Parse with regex: `(Monday|Tuesday|...|Sunday)\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM)\s*-\s*\d{1,2}(?::\d{2})?\s*(?:AM|PM)|Closed)`

### 3. `.socialProfile` element (social media)

Contains `<a>` links with platform names as text and full profile URLs as href.

### 4. `data-facts` JSON attribute (structured address)

```json
{
  "addressFields": {
    "addressLine": "708 Nueces St",
    "city": "Austin",
    "stateMunicipality": "TX",
    "postalCode": "78701"
  }
}
```

### 5. `data-entitymetadata` JSON attribute

Contains `PCID` (Bing category ID), `PivLL` (lat/lng), and segment type.

## Parsing Fallbacks (Discovery Mode)

1. **`data-entity`** (primary) — JSON with entity + geometry
2. **`data-bm`** — alternate JSON format on some variants
3. **`.b_scard` containers** — CSS-based extraction
4. **`#taskPaneBody .b_entityTP`** — sidebar entity panels

## Other Useful data-* Attributes

| Attribute | Value | Use |
|-----------|-------|-----|
| `data-hasmoreresults` | `true`/`false` | Discovery pagination signal |
| `data-iscategoryquery` | `true`/`false` | Whether query maps to a category |
| `data-ig` | hex GUID | IG session token |
| `data-queryparse` | JSON | Bing's query interpretation |
| `data-itineraryfacts` | JSON | Structured address + nav data |
| `data-streetside` | JSON | Streetside view coordinates |

## Performance

- Discovery (~10 results): ~147KB response
- Detail (~1 result): ~330KB response
- Full Maps page: ~2MB (JS-rendered shell, data loads via AJAX)
- Pure HTTP — works with curl_cffi or any HTTP client
- No JavaScript execution needed
- Supports IPv6 rotation
- Pagination via `pos` parameter (offset-based)

## Workflow for Large-Scale Extraction

```
1. Discovery sweep: 40k zip codes × category queries × paginate
   → Collect roster: name, address, phone, website, coords, ypid, basic rating
   → ~14 fields per venue, 10 venues per call

2. Detail enrichment: For each discovered venue, query by name+city+state
   → Full detail: structured reviews, hours, description, social media, photos
   → ~22 fields per venue, 1 venue per call

3. (Optional) Website enrichment: Follow venue website URLs
   → Run GhostGraph schema cascade for domain-specific data
   → Capacity, pricing, amenities, full photo galleries
```

Phase 2 does NOT require visiting venue websites — the overlay endpoint itself
returns the detail panel when narrowed to a single entity. This is a huge win:
same proxy rotation, same endpoint, consistent data format.

Phase 3 is optional and only needed for data that Bing doesn't aggregate
(venue-specific pricing, capacity, detailed amenity lists).
