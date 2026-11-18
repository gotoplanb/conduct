# Voice cast

Conduct's voice registry (`/voices/registry`, #51) is the map of every voice
a client can ask for by name. This page describes what's in it, how the
voices are meant to be used, and how to cast a multi-voice story.

Conduct owns the **voices**. The **cast**, meaning which character gets which voice,
lives in the client (Wander, the Bosshardt demo director, …). A client sends
`{"text": ..., "voice": "<alias>"}` to `/tts`, and Conduct resolves the alias to an
engine and a concrete voice. See `docs/tts.md` for the API itself.

## Tiers

| Tier | Engine | Over the API? | Cost | Data leaves the box? |
|---|---|---|---|---|
| Local | `piper` | Yes | Free | No |
| ElevenLabs premade | `elevenlabs` | Yes, any plan | $0.08 / 1K chars (v2 Multilingual) | Yes |
| ElevenLabs library (community / professional) | `elevenlabs` | **Paid plan only** (Starter+). Free plan returns `402 paid_plan_required` | Same, plus library terms | Yes |
| ElevenLabs Iconic (™ celebrity voices, e.g. Sir Michael Caine) | n/a | **No.** Returns `401 famous_voice_not_permitted`. ElevenReader app only; API/commercial use needs a per-use license approved by the rights holder | n/a | n/a |

Don't register Iconic voices: they will fail at synthesis time. If a story
wants "a Michael Caine", cast the closest premade (`el-george` or
`el-daniel`) and note the intent in the client's cast.

ElevenLabs aliases need the calling client to have an ElevenLabs key
(`conduct clients set-elevenlabs-key <client>`). Sensitivity gating applies:

- `public`: allowed
- `internal`: needs the client's `allow_cloud_for_internal`
- `confidential`: always refused

Piper aliases are unaffected by any of this.

## Naming

- `el-<name>`: a raw ElevenLabs voice, named after the voice itself
  (`el-george`). Stable; don't repoint these to a different voice.
- Role names (`narrator`, `first-responder`, `bosshardt-narrator`): what the
  voice is *for*. These are the ones to repoint when recasting. Changing the
  row recasts every client using the role without a client deploy.
- Per-client overrides: a row with `client_id` set shadows the global row of
  the same name for that client only.
- Clients that cache audio by `(text, voice name)` (e.g. the Bosshardt
  director) won't notice a repoint. Use a new role name, or clear the cache,
  when you recast.

## Roster

### Local (Piper)

| Alias | Voice | Use |
|---|---|---|
| `narrator` | en_US-amy-medium | Default long-form narrator |
| `ops-manager` | en_GB-alan-medium | Second voice for multi-participant scenes |
| `first-responder` | en_US-ryan-high | Tier 1 responder (#54) |
| `escalation-desk` | en_US-joe-medium | Tier 2 desk (#54) |
| `customer` | en_GB-jenny_dioco-medium | External customer, pairs with Perform phone treatment (#54) |
| `feedback-coach` | en_US-lessac-medium | Out-of-world coaching voice (#54) |

### ElevenLabs premade

All 21 are registered globally as `el-<name>`.

| Alias | Sound | Accent | Good for |
|---|---|---|---|
| `el-george` | Warm, captivating storyteller | British, middle-aged M | Story narrator; the closest premade to a Caine-style narrator |
| `el-daniel` | Steady broadcaster, formal | British, middle-aged M | Documentary narration, newsreader, authority figure |
| `el-alice` | Clear, engaging educator | British, middle-aged F | Explainers, tutorial narration, mentor |
| `el-lily` | Velvety actress, confident | British, middle-aged F | Lead character, dramatic narration |
| `el-matilda` | Knowledgeable, professional, upbeat | American, middle-aged F | Product narration, walkthroughs |
| `el-bella` | Professional, bright, warm | American, middle-aged F | Product narration, friendly guide |
| `el-eric` | Smooth, trustworthy | American, middle-aged M | Product narration, spokesperson |
| `el-chris` | Charming, down-to-earth | American, middle-aged M | Casual narration, everyman character |
| `el-brian` | Deep, resonant, comforting | American, middle-aged M | Trailer voice, calm authority |
| `el-bill` | Wise, mature, crisp | American, older M | Elder, mentor, ad read |
| `el-river` | Relaxed, neutral, informative | American, neutral | Neutral narrator, system/AI voice |
| `el-roger` | Laid-back, resonant | American, middle-aged M | Conversational character |
| `el-sarah` | Mature, reassuring, confident | American, young F | Protagonist, reassuring guide |
| `el-jessica` | Playful, bright, warm | American, young F | Sidekick, upbeat character |
| `el-laura` | Enthusiastic, quirky, sassy | American, young F | Comic relief, social clips |
| `el-will` | Relaxed optimist | American, young M | Friendly sidekick |
| `el-liam` | Energetic creator | American, young M | Hype, social clips |
| `el-charlie` | Deep, confident, energetic | Australian, young M | Adventurer, outsider character |
| `el-adam` | Dominant, firm | American, middle-aged M | Antagonist, commander |
| `el-callum` | Husky trickster | American, middle-aged M | Rogue, trickster, villain |
| `el-harry` | Fierce warrior, rough | American, young M | Warrior, action character |

## Casting a story

1. **Pick a narrator first.** It carries most of the characters, so choose for
   long-form comfort, not novelty. The ElevenLabs model
   (`eleven_multilingual_v2`) is tuned for long-form stability.
2. **One voice per character, for the whole work.** The client's cast maps
   `character → alias`. Never vary the voice mid-story to signal emotion;
   that's the text's job.
3. **Contrast before character.** Characters who share scenes should differ on
   at least two of: gender, accent, age, energy. `el-george` + `el-alice`
   (both British, middle-aged) is fine across gender, but `el-george` + `el-daniel`
   in the same scene will blur.
4. **Keep out-of-world voices out of the cast.** Coaching, system prompts and
   UI voices (`feedback-coach`, `el-river`) shouldn't also play a character.
5. **Mix engines deliberately.** Piper is free and local; ElevenLabs is
   richer but costs per character and leaves the box. A reasonable split is
   ElevenLabs for the narrator and leads, and Piper for bit parts and drafts.
   Draft the whole story on Piper role names, then repoint the roles to
   ElevenLabs for the final render.
6. **Budget.** At $0.08 per 1K characters, a 10K-character chapter is about $0.80. Job
   cost lands on `jobs.cost_usd` and the client's usage rollup.

## Adding a voice

```bash
curl -X PUT http://localhost:8000/voices/registry/<alias> \
  -H "Authorization: Bearer $CONDUCT_ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"engine":"elevenlabs","voice_file":"<voice_id>","notes":"<sound, accent, use>"}'
```

- **ElevenLabs:** the voice ID is on the voice's page in the ElevenLabs app,
  or comes from `GET /v1/voices`. A library voice must first be added to the
  account, and needs a paid plan to synthesize via the API.
- **Piper:** `make download-voice v=<stem>`, then register with
  `"engine":"piper","voice_file":"<stem>"`.

Then add a row to the roster above.
