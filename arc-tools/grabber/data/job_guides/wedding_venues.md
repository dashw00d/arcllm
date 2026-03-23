# Wedding Venues Campaign Guide

You are orchestrating a nationwide wedding/event venue data collection campaign using the GhostGraph API. This guide describes the strategy. You make decisions dynamically based on results.

Base URL: `http://localhost:8000` (or the configured API host)

---

## Phase 1: Bing Maps Search by Zip Code

Goal: Seed the system with venue listings discovered via Bing Maps search.

### Setup

1. Create a project:
```
POST /api/projects
{
  "name": "Wedding Venues US",
  "job_type": "wedding_venues",
  "description": "Nationwide wedding and event venue data collection"
}
```
Save the returned `id` as your `project_id`.

2. Load zip codes from `data/us_zipcodes.csv`. Format:
```
zip,city,state,lat,lng,population
99553,Akutan,AK,54.143,-165.7854,0
10001,New York,NY,40.7506,-73.9971,21102
```

### Strategy

- Sort zip codes by population descending. High-population zips yield more venues.
- Skip zips with population=0 (remote areas with no venues).
- Start with a pilot batch of ~1000 high-population zips to validate quality before scaling.
- Use Bing Maps search URLs as the task URL.

### Search Queries

Effective query patterns (pick one per zip, or rotate):
- `wedding venues near {zip}`
- `wedding venues {city}, {state}`
- `event venues near {zip}`
- `banquet halls {city}, {state}`
- `reception halls near {zip}`

### Submitting Tasks

Submit in batches of 100 (API limit per request):
```
POST /api/tasks/batch
{
  "tasks": [
    {
      "url": "https://www.bing.com/maps?q=wedding+venues+near+10001",
      "entity_type": "venue_listing",
      "project_id": "<project_id>"
    },
    ...
  ]
}
```

Notes:
- Use `entity_type: "venue_listing"` for this phase (basic listings, not full venue profiles).
- If you get HTTP 429, the stream is at capacity. Wait 30-60 seconds and retry.
- The response includes a `count` field showing how many were actually created. If it is less than submitted, backpressure kicked in.

### Monitoring

```
GET /api/projects/<project_id>/summary
```

Returns task state breakdown (pending/exploring/scheming/extracting/completed/failed) and entity count.

```
GET /api/dashboard
```

Returns system-wide view including stream backlog and worker utilization.

### Expected Output

Each completed task produces entities with fields like:
- name, address, city, state, zip_code, phone, website, rating, review_count

### Budget

- ~40,000 zips with population > 0
- At ~$0.01/task LLM cost: ~$400 for full coverage
- Pilot 1000 zips first: ~$10

---

## Phase 2: Identify Aggregator Sites

Goal: Find aggregator domains that list many venues. These are high-value targets for bulk scraping.

### After Phase 1 Completes

Check entity statistics:
```
GET /api/entities/stats
```

Response includes `by_domain`: a list of source domains with entity counts.

### Known Aggregator Domains

These are the major wedding venue aggregator sites. If they appear in your results, prioritize them:

**Tier 1 (largest directories):**
- theknot.com
- weddingwire.com
- zola.com
- wedding.com

**Tier 2 (specialty directories):**
- wedding-spot.com / weddingspot.com
- herecomestheguide.com
- partyslate.com
- tagvenue.com
- eventup.com
- venues.com / weddingvenues.com
- peerspace.com
- giggster.com
- uniquevenues.com

**Tier 3 (editorial/curated):**
- junebugweddings.com
- caratsandcake.com
- brides.com
- stylemepretty.com
- weddingrule.com

**General directories (venue data often present):**
- yelp.com
- tripadvisor.com
- yellowpages.com
- eventective.com

### Dynamic Discovery

Any domain appearing 50+ times in your entity stats is likely an aggregator even if not listed above. Flag it for deep scraping.

### Decision Logic

For each aggregator found:
1. Check if GhostGraph already has a pattern for that domain: `GET /api/patterns/domain/{domain}`
2. If no pattern exists, the system will generate one automatically during the explore/schema pipeline.
3. Create a task for the aggregator's venue listing/search page.

---

## Phase 3: Deep Scrape Aggregators

Goal: Submit aggregator listing pages as tasks. GhostGraph's explore-schema-extract pipeline handles the rest.

### Submitting Aggregator Tasks

For each aggregator, submit its venue listing or search page:
```
POST /api/tasks
{
  "url": "https://www.theknot.com/marketplace/wedding-reception-venues",
  "entity_type": "venue",
  "project_id": "<project_id>"
}
```

Use `entity_type: "venue"` (not "venue_listing") -- this tells the schema generator to extract full venue profiles.

### What the Pipeline Does

1. **Explore**: Discovers all venue detail page URLs from the listing page (pagination, links, sitemaps).
2. **Schema**: Samples 3 detail pages, generates an extraction schema for that domain.
3. **Extract**: Applies the schema to every discovered detail page, creating one entity per venue.

### Expected Fields

The schema generator will look for these fields (guided by `entity_type: "venue"`):
- name, address, city, state, zip_code
- phone, email, website
- rating, review_count
- capacity (min/max guests)
- price_range / starting_price
- amenities (list)
- venue_type (barn, estate, hotel, garden, etc.)
- photos (list of URLs)
- description

### Monitoring

```
GET /api/projects/<project_id>/summary
```

Watch for:
- `task_states.exploring > 0` means workers are discovering pages
- `task_states.scheming > 0` means schema generation is running
- `task_states.extracting > 0` means extraction is in progress
- `entity_count` should be growing

Check for failures:
- `recent_failures` shows the last 10 failed tasks with `error_message` and `failed_stage`
- If many tasks fail at `exploring`, the site may be blocking. Consider waiting or adjusting.
- If many tasks fail at `scheming`, the site structure may be unusual. Check a sample page manually.

### Scaling

Check worker capacity:
```
GET /api/dashboard
```

Look at:
- `stream_backlog`: if queues are growing faster than workers can process, deploy more workers.
- `worker_summary.active`: number of workers currently processing.

Deploy more workers:
```
POST /api/fleet/deploy
{"count": 5}
```

---

## Phase 4: Individual Venue Enrichment

Goal: Fill in missing data for venues that lack key fields.

### Finding Incomplete Venues

Search for venues missing website or phone:
```
POST /api/entities/search
{
  "query": "wedding",
  "entity_type": "venue"
}
```

Then filter client-side for entities where `data.website` or `data.phone` is null/missing.

### Enrichment Strategy

For each incomplete venue:
1. Search Bing for: `"{venue_name}" {city} {state} wedding venue`
2. The top result is often the venue's own website.
3. Submit the venue's website as a task:
```
POST /api/tasks
{
  "url": "https://www.example-venue.com",
  "entity_type": "venue",
  "project_id": "<project_id>"
}
```

### Priority

Focus enrichment on venues that have:
- A name and address (confirmed real)
- Missing website, phone, capacity, or pricing
- Appeared in multiple aggregator sources (likely a real popular venue)

---

## Phase 5: Entity Deduplication and Merge

Goal: Merge duplicate venue records from different sources into a single canonical entity.

### Finding Duplicates

By address (strongest signal):
```
POST /api/entities/find-duplicates
{
  "project_id": "<project_id>",
  "entity_type": "venue",
  "match_fields": ["address"],
  "threshold": 0.8
}
```

By name + city (catches slight address variations):
```
POST /api/entities/find-duplicates
{
  "project_id": "<project_id>",
  "entity_type": "venue",
  "match_fields": ["name", "city"],
  "threshold": 0.8
}
```

Response: groups of potential duplicates with a `confidence` score.

### Merging

For each duplicate group:
```
POST /api/entities/merge
{
  "entity_ids": ["<id1>", "<id2>", "<id3>"],
  "primary_id": "<id_from_own_website>"
}
```

Merge rules:
- Set `primary_id` to the entity from the venue's own website (most authoritative data).
- If no entity is from the venue's own website, prefer the entity with the most populated fields.
- The merge deep-merges JSONB data: primary values win for scalars, lists are unioned.
- Merged-away entities get `status: "merged"` and are excluded from future queries.

### Quality Check

After merging, check overall data quality:
```
GET /api/entities?entity_type=venue&status=active&limit=500
```

Assess what percentage of venues have:
- name + address: should be ~100%
- phone: target 70%+
- website: target 60%+
- capacity or price_range: target 40%+

---

## Domain Exclusions

Never scrape these domains. They are search engines, social media, news sites, or otherwise not useful for venue data.

**Search engines:** google.com, bing.com, yahoo.com, duckduckgo.com

**Social media:** facebook.com, instagram.com, pinterest.com, linkedin.com, twitter.com, x.com, tiktok.com, youtube.com, reddit.com

**News/media:** cnn.com, bbc.com, nytimes.com, npr.org, forbes.com, huffpost.com, msn.com, washingtonpost.com, foxnews.com, nbcnews.com, cbsnews.com, usatoday.com, time.com, businessinsider.com, cnbc.com

**E-commerce:** amazon.com, walmart.com, target.com, ebay.com, etsy.com

**Travel/booking (no structured venue data):** expedia.com, booking.com, hotels.com, trivago.com, kayak.com, airbnb.com, vrbo.com

**Job/real estate sites:** indeed.com, glassdoor.com, ziprecruiter.com, zillow.com, realtor.com, redfin.com, trulia.com, apartments.com

**Ticket/event platforms:** ticketmaster.com, stubhub.com, eventbrite.com, meetup.com

**Reference/tools:** wikipedia.org, stackoverflow.com, github.com, medium.com, wordpress.com, blogspot.com

**Non-US TLDs:** Skip domains ending in .de, .uk, .co.uk, .fr, .es, .it, .nl, .ru, .jp, .cn, .br, .mx, .au, .ca, and other non-.com/.org/.net TLDs. This is a US-focused campaign.

This list is guidance, not exhaustive. If a domain clearly has no venue listing data, skip it.

---

## Retry and Error Handling

- **Task state `failed`**: Check `failed_stage` and `error_message` via `GET /api/tasks/<id>`.
  - `failed_stage: "exploring"`: Site may be blocking or down. Wait and retry: `POST /api/tasks/<id>/retry`
  - `failed_stage: "scheming"`: Schema generation failed. The site may have unusual structure. Check manually before retrying.
  - `failed_stage: "extracting"`: Circuit breaker may have tripped (60%+ work items failing). Check work items: `GET /api/tasks/<id>/work-items?state=failed`
- **HTTP 429 on task submission**: Stream backpressure. Wait 30-60 seconds.
- **HTTP 409 on cancel/retry**: Task is already in a terminal state (completed/failed/cancelled).

---

## Workflow Summary

```
1. Create project
2. Submit Bing Maps search tasks (1000 pilot, then scale)
3. Monitor until Phase 1 completes
4. Analyze entity stats to find aggregator domains
5. Submit aggregator listing pages as tasks
6. Monitor until Phase 3 completes
7. Identify venues with missing data
8. Submit enrichment tasks for venue websites
9. Run deduplication and merge
10. Quality check final dataset
```

Each phase depends on the previous one completing. Monitor via `/api/projects/<id>/summary` and `/api/dashboard`. Scale workers up via `/api/fleet/deploy` when queues grow, scale down via `/api/fleet/destroy` when work is done.
