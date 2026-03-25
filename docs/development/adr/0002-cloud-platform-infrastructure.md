# ADR-0002: Cloud Platform Infrastructure

| Field         | Value                                                                                       |
|---------------|---------------------------------------------------------------------------------------------|
| **Status**    | Proposed                                                                                    |
| **Date**      | 2026-03-24                                                                                  |
| **Review By** | 2026-09-24 (6 months post-adoption)                                                         |
| **Deciders**  | JABS Hub Development Team                                                                   |
| **Category**  | Infrastructure & Deployment                                                                 |
| **Scope**     | Development environment; production sizing and hardening to be revisited in a follow-up ADR |

---

## Context

JABS Hub is a cloud-hosted platform that centralizes the storage and management of JABS
behavioral annotation projects. It exposes an authenticated HTTP API consumed by both a
web UI (Angular) and the JABS desktop GUI client. The platform must support video and
pose file storage, structured label management, project metadata, and future integration
with Envision.

The Kumar Lab's current workflow relies on local file-based JABS projects, which
introduces pain points around collaboration, version control, data duplication, and
remote access. JABS Hub addresses these by moving project data and metadata into the
cloud while preserving the desktop GUI as the primary annotation interface.

Key constraints and requirements driving this decision include:

- **Small user base, growing data volume.** The Kumar Lab has a relatively small number
  of concurrent users, but the volume of stored videos, pose files, annotations, and
  classifier artifacts will grow substantially over time.
- **Managed services preferred.** The team is small (2–3 developers) and cannot afford
  to operate and maintain custom infrastructure. Managed services reduce operational
  burden.
- **Institutional alignment.** The team has prior experience deploying FastAPI
  applications on GCP with Cloud Run, and the JDS team provides an Angular-based UI
  toolkit aligned with the project's design system.
- **Auth0 for authentication.** The organization uses Auth0 as its identity provider,
  supporting OIDC-based authentication flows.
- **Cost sensitivity.** As a research platform, JABS Hub must keep cloud costs
  reasonable and predictable, with per-video cost visibility where possible.
- **Future extensibility.** The architecture must allow natural evolution toward
  multi-tenant use, Envision integration, and potential adoption by research groups
  beyond the Kumar Lab.

This ADR records the decision on cloud provider, compute platform, frontend hosting,
object storage, database, infrastructure-as-code tooling, and CI/CD pipeline for the
JABS Hub MVP.

**Scope note:** This ADR targets the **development environment** and establishes the
service selections and architectural patterns that will carry forward into staging and
production. Configuration details such as instance sizing, concurrency limits, high
availability, connection pooling, and disaster recovery targets are set here for a
lightweight dev workload and will be revisited in a production-readiness ADR before the
platform is opened to end users.

**Data residency and compliance:** JABS Hub stores mouse behavioral video data, pose
estimation outputs, and researcher-authored annotations. No protected health
information (PHI), personally identifiable information (PII), or regulated data is
stored in the platform. Researcher identity metadata (names, email addresses) is managed
by Auth0 and is not replicated into JABS Hub's database beyond user IDs and display
names. If future Envision integration or institutional policy changes introduce data
locality or compliance requirements, this decision should be revisited. For now,
standard GCP data handling practices apply with no additional regulatory constraints.

---

## Decision

We will deploy JABS Hub on **Google Cloud Platform (GCP)** using the following managed
service architecture:

| Component                  | Service                                          | Rationale                                                     |
|----------------------------|--------------------------------------------------|---------------------------------------------------------------|
| **Backend API**            | Cloud Run (FastAPI in a container)               | Serverless containers, scales to zero, minimal ops            |
| **Frontend Web UI**        | Cloud Storage + Cloud CDN (Angular static build) | Static hosting with CDN edge caching, full GCP-native control |
| **Object Storage**         | Google Cloud Storage (Standard tier)             | Video, pose files, and derived data; pre-signed URL support   |
| **Metadata Database**      | Cloud SQL for PostgreSQL (Enterprise edition)    | Managed relational DB for projects, labels, audit history     |
| **Authentication**         | Auth0 (OIDC)                                     | Institutional standard; issues JWTs validated by the API      |
| **Infrastructure as Code** | Terraform                                        | Industry-standard, strong GCP provider, declarative           |
| **CI/CD**                  | GitHub Actions                                   | Existing team tooling; native container build and deploy      |
| **Container Registry**     | Google Artifact Registry                         | Native integration with Cloud Run and Cloud Build             |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Clients                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│   │ Angular Web  │    │ JABS Desktop │    │  Future Clients  │  │
│   │ UI (Browser) │    │ GUI (Python) │    │  (Envision, etc) │  │
│   └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘  │
└──────────┼───────────────────┼─────────────────────┼────────────┘
           │                   │                     │
           ▼                   ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  Cloud Storage + Cloud CDN                             │      │
│  │  (Angular static assets, global edge caching)          │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  ┌────────────────────┐      ┌─────────────────────────────┐     │
│  │ Cloud Run          │◄────►│ Cloud SQL (PostgreSQL)      │     │
│  │ (FastAPI Backend)  │      │ Projects, labels, metadata  │     │
│  └────────┬───────────┘      └─────────────────────────────┘     │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────────────────────────┐          │
│  │  Google Cloud Storage (Standard)                   │          │
│  │  Videos (.mp4), Pose files (.h5), Derived data     │          │
│  │  Pre-signed URLs for secure client download        │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                  │
│  ┌────────────────────┐      ┌─────────────────────────────┐     │
│  │ Auth0 (OIDC)       │      │  Artifact Registry          │     │
│  │ Identity & Tokens  │      │  Container images           │     │
│  └────────────────────┘      └─────────────────────────────┘     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Detailed Decisions

### 1. Cloud Provider: Google Cloud Platform

**Decision:** GCP

**Why:** The team has direct experience deploying FastAPI + Angular on GCP with Cloud
Run. GCP's managed services (Cloud Run, Cloud SQL, GCS) align well with the team's size
and operational capacity. The organization can provision a dedicated GCP project for
JABS Hub. GCP's networking model provides free data transfer between services in the
same region, which is important given the volume of video data flowing between Cloud
Storage and Cloud Run.

### 2. Backend Compute: Cloud Run

**Decision:** Cloud Run with request-based billing

**Why:** Cloud Run provides serverless container hosting with automatic scaling,
including the ability to scale to zero during periods of inactivity. The FastAPI
application is stateless and container-friendly. Cloud Run's request-based billing model
means the team only pays for actual compute during request handling, which is
cost-efficient for a platform with a small, bursty user base. Cloud Run also supports
concurrency (multiple requests per instance), which improves efficiency for API
workloads.

**Configuration guidance (development environment):**

- Region: `us-east4` (or the region closest to the primary user base)
- Min instances: 0 (scale to zero for cost savings; accept cold starts)
- Max instances: 5 (sufficient for dev; increase for staging/production)
- CPU: 1 vCPU, Memory: 512 MiB (start small, right-size based on monitoring)
- Concurrency: 20–40 requests per instance (start conservative; tune based on p99
  latency under load — endpoints that generate signed URLs or query the database may not
  benefit from higher concurrency on a single vCPU)
- Cloud Run connects to Cloud SQL via the built-in Cloud SQL Auth Proxy sidecar (
  recommended) or Unix socket connector

**Connection pooling:** Cloud Run autoscaling can open many concurrent database
connections. SQLAlchemy's connection pool should be configured with a `pool_size` and
`max_overflow` that, when multiplied by the max instance count, stays within Cloud SQL's
connection limit (~100 for `db-custom-1-3840`). For example,
`pool_size=5, max_overflow=5` with max 5 instances = 50 potential connections, well
within limits. For production scaling, introduce PgBouncer as a sidecar or use Cloud
SQL's managed connection pooling feature.

### 3. Frontend Hosting: Cloud Storage + Cloud CDN

**Decision:** Google Cloud Storage bucket with Cloud CDN and a Cloud Load Balancer for
the Angular web UI

**Why:** The Angular frontend compiles to static assets (HTML, CSS, JS) that do not
require server-side rendering. A GCS bucket configured for static website hosting,
fronted by Cloud CDN via a global HTTP(S) load balancer, provides edge caching,
automatic SSL via Google-managed certificates, and full control over caching, routing,
and security headers — all within the GCP ecosystem and manageable through Terraform.

**Setup requirements:**

- GCS bucket configured with `MainPageSuffix: index.html` and
  `NotFoundPage: index.html` (the latter enables Angular's client-side routing — all
  paths resolve to `index.html` so the Angular router handles them)
- Global external HTTP(S) load balancer with the GCS bucket as a backend
- Google-managed SSL certificate on the load balancer for the custom domain
- Cloud CDN enabled on the backend bucket with appropriate cache TTLs (long TTLs for
  hashed asset files, short for `index.html`)
- **CORS configuration** on the API (Cloud Run): The Angular SPA on a separate domain
  will make cross-origin requests to the FastAPI backend. The API must return
  appropriate `Access-Control-Allow-Origin`, `Access-Control-Allow-Headers`, and
  `Access-Control-Allow-Methods` headers. FastAPI's `CORSMiddleware` handles this.

**Deployment:** `gsutil rsync` or `gcloud storage cp` from the Angular build output
directory to the bucket, followed by a CDN cache invalidation for `index.html`. This
integrates cleanly into the GitHub Actions pipeline.

**Alternatives considered for frontend hosting:**

| Option                                 | Pros                                                                                                     | Cons                                                                                                                      |
|----------------------------------------|----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **Cloud Storage + Cloud CDN** (chosen) | Full GCP-native control, Terraform-managed, CDN edge caching, Google-managed SSL, flexible caching rules | Requires load balancer setup (one-time Terraform config); CDN invalidation on deploy                                      |
| **Firebase Hosting**                   | Simpler CLI deploy, atomic rollback, generous free tier                                                  | Separate billing/project model from core GCP infra; less Terraform control; adds Firebase dependency for a single feature |
| **Cloud Run (nginx container)**        | Consistent with backend hosting model                                                                    | Overkill for static assets; pay for idle compute; slower deploys                                                          |

### 4. Object Storage: Google Cloud Storage (Standard)

**Decision:** GCS Standard tier in a single regional bucket

**Why:** All videos, pose estimation files (.h5), and derived data (features,
predictions, classifier artifacts) are stored in GCS. The FastAPI backend generates
time-limited pre-signed (signed) URLs that allow the JABS GUI and web UI to download
media directly from GCS without proxying through the API. This is a standard pattern for
large-file access that avoids bottlenecking the API service.

**Bucket design:**

- Single regional bucket (same region as Cloud Run and Cloud SQL)
- Bucket naming convention: `jabs-hub-{env}-data` (e.g., `jabs-hub-prod-data`)
- Object key structure: `projects/{project_id}/videos/`, `projects/{project_id}/pose/`,
  `projects/{project_id}/classifiers/`
- Lifecycle rules: consider Nearline transition for objects not accessed in 90+ days (
  future optimization)
- Versioning: enabled for data integrity; lifecycle rule to delete old versions after a
  retention period

**Pre-signed URL security model:**

- Only authenticated users with project-level access (verified by the API against Auth0
  token claims) can request signed URLs. The API checks project membership before
  generating any URL.
- Signed URLs are time-limited: **1 hour TTL for downloads**, **30 minute TTL for
  uploads**. TTLs are configurable and should be kept as short as practical.
- **Downloads** use signed URLs generated by the API; the JABS GUI and web UI fetch
  media directly from GCS.
- **Uploads** also use signed URLs (resumable upload type for large video files). The
  client uploads directly to GCS, then notifies the API to register the object metadata.
  This avoids proxying large files through Cloud Run.
- All signed URL generation events are logged (user, project, object key, timestamp) in
  the audit table for traceability. Note that once a signed URL is issued, access to the
  object bypasses Auth0 for the duration of the TTL. Keep TTLs short and log generation
  events.

**Data migration from local projects:** Existing JABS projects stored as local
directories will be migrated into JABS Hub via a CLI migration tool (to be built as part
of the Phase 2 development effort). The tool will authenticate via Auth0, create a Hub
project, upload videos and pose files to GCS via signed URLs, and register metadata via
the API. Manual migration by individual researchers is also supported through the web
UI's upload workflow. A detailed migration runbook will be produced alongside the tool.

**Storage abstraction — MinIO and GA4GH DRS:**

The team has discussed the possibility of using MinIO as a local development and testing
fixture, and the GA4GH Data Repository Service (DRS) pattern as a storage abstraction
layer. Both have merit:

- **MinIO** provides an S3-compatible API that can serve as a local development stand-in
  for GCS, enabling integration tests without cloud access and providing an escape hatch
  if the project later migrates away from GCP. GCS's own S3-compatible interoperability
  mode is an alternative, but MinIO gives full local control. **Recommendation:** Use
  MinIO in the local development and CI test environments. This adds minimal overhead
  and provides valuable portability.

- **GA4GH DRS** provides durable URIs and a resolution layer that abstracts across
  storage backends (GCS, S3, on-prem). It was designed for cross-institutional data
  sharing in genomics but is conceptually applicable to any large-file storage. However,
  standing up a DRS service adds architectural complexity and an additional service to
  maintain. For the MVP, the team should **code against a thin storage interface** (
  e.g., a Python ABC with `generate_signed_url()`, `upload()`, `exists()` methods)
  backed by the GCS client library. This keeps the door open for DRS or backend
  migration later without introducing premature abstraction. **Recommendation:** Defer
  DRS to a post-MVP evaluation. Use a thin abstraction interface in the API layer so
  storage backends can be swapped without major refactoring.

### 5. Database: Cloud SQL for PostgreSQL

**Decision:** Cloud SQL Enterprise edition, PostgreSQL 16

**Why:** The JABS Hub data model (projects, videos, labels, video-project associations,
audit events, classifier metadata) is inherently relational with many-to-many
relationships and requires transactional consistency for label updates. PostgreSQL is
well-suited for this, and Cloud SQL provides automated backups, patching, point-in-time
recovery, and high availability options without requiring DBA expertise.

**Configuration guidance (development environment):**

- Instance: `db-custom-1-3840` (1 vCPU, 3.75 GiB RAM) — adequate for development and
  light testing; will be re-evaluated for production based on observed query patterns
  and connection load
- Storage: 10 GB SSD, with auto-resize enabled
- Backups: automated daily, 7-day retention (sufficient for dev; production will define
  explicit RTO/RPO targets)
- High availability: single-zone for dev; regional HA to be enabled before production
  launch

**Networking:** Cloud Run requires a Serverless VPC Access connector (or Direct VPC
egress) to reach a Cloud SQL instance on a private IP. The VPC connector is provisioned
in the same region and incurs a small cost (~$7/mo for an `e2-micro` connector). The
Cloud SQL instance exposes no public IP; all access is through the private VPC path. The
connector must be specified in the Cloud Run service configuration and provisioned via
Terraform.

**Secret management:** Database credentials, Auth0 client secrets, and GCS service
account keys are stored in **Google Secret Manager**. Cloud Run services access secrets
at startup via Secret Manager volume mounts or environment variable injection (
configured in the Cloud Run service definition). No secrets are stored in environment
variables in the Terraform state, GitHub Actions secrets, or application code.

**Database migrations:** The project will use **Alembic** for schema migrations,
integrated into the CI/CD pipeline. Migrations run as a pre-deploy step in GitHub
Actions (via a Cloud Run job or a direct `alembic upgrade head` against the dev database
through Cloud SQL Auth Proxy). Migration scripts are version-controlled alongside the
FastAPI application code.

### 6. Infrastructure as Code: Terraform

**Decision:** Terraform with the Google Cloud provider

**Why:** Terraform is the industry standard for declarative infrastructure management.
It has mature support for all GCP services used in this architecture (Cloud Run, Cloud
SQL, GCS, Firebase Hosting, VPC, IAM). The team can version infrastructure definitions
alongside application code in the same repository. Terraform's plan/apply workflow
provides safe, reviewable infrastructure changes.

**Alternatives considered:**

| Tool                                | Pros                                                                                     | Cons                                                                        |
|-------------------------------------|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Terraform** (chosen)              | Industry standard, large community, strong GCP support, declarative                      | HCL learning curve (modest)                                                 |
| **Pulumi**                          | Use real programming languages (Python, TS), good for teams that prefer imperative style | Smaller community, state management complexity, less GCP ecosystem adoption |
| **Google Cloud Deployment Manager** | Native to GCP, no external tooling                                                       | GCP-only (vendor lock-in for IaC), limited community, less flexible         |

**Recommendation:** Use Terraform. If the team later decides to manage infrastructure
across multiple clouds or integrate with other institutional systems, Terraform's
provider ecosystem makes that straightforward.

### 7. CI/CD: GitHub Actions

**Decision:** GitHub Actions for build, test, and deploy pipelines

**Why:** The team already uses GitHub for source control. GitHub Actions provides native
integration with container builds (Docker), GCP authentication (via Workload Identity
Federation), and deployment to Cloud Run and Firebase Hosting. There is no need to
introduce a separate CI/CD platform.

**Pipeline structure:**

- **On push to `main`:** Build container → push to Artifact Registry → run Alembic
  migrations → deploy to Cloud Run (staging) → run integration tests → deploy Angular
  build to GCS bucket (staging) → invalidate CDN cache
- **On release tag:** Promote staging images/builds to production
- **On pull request:** Run unit tests, linting, type checks

### 8. Authentication: Auth0

**Decision:** Auth0 as the OIDC identity provider

**Why:** Auth0 is the organization's standard identity platform. The FastAPI backend
will validate Auth0-issued JWTs on every request. The Angular web UI and JABS desktop
GUI will both use Auth0's OIDC authorization code flow (with PKCE for public clients) to
obtain tokens. Auth0 provides user identity, group/role claims, and supports MFA if
required.

**Integration points:**

- Angular web UI: Auth0 SPA SDK for login/logout, token management
- JABS GUI (Python desktop): Auth0 device authorization flow or PKCE-based browser
  redirect
- FastAPI: `authlib` for JWT validation; extract user identity and roles from token
  claims
- API authorization: role-based access control using Auth0 custom claims (e.g.,
  project-level permissions)

### 9. Observability: Cloud Logging + Cloud Monitoring

**Decision:** Use GCP's native observability stack

**Why:** Cloud Run, Cloud SQL, and GCS all emit logs and metrics to Cloud Logging and
Cloud Monitoring by default. For a small team, using the built-in stack avoids
introducing additional services while still providing sufficient visibility.

**Setup:**

- **Structured logging:** The FastAPI application will emit JSON-formatted logs (using
  Python's `structlog` or `logging` with a JSON formatter) so that Cloud Logging can
  parse and filter by request ID, user ID, endpoint, and latency.
- **Uptime checks:** Cloud Monitoring uptime checks configured against the API's
  `/health` endpoint, with alerting to a shared team email or Slack channel.
- **Cloud SQL metrics:** Monitor CPU utilization, connection count, disk usage, and
  query latency via Cloud Monitoring dashboards.
- **Alerting:** Basic alert policies for Cloud SQL connection saturation (>80% of max
  connections), Cloud Run error rate spikes, and GCS egress anomalies.
- **Future consideration:** If the team needs distributed tracing across services, Cloud
  Trace integration is available with minimal code changes. Error reporting via Sentry
  is an option if GCP's native error reporting is insufficient.

### 10. API Security: Cloud Run Default Ingress

**Decision:** Use Cloud Run's default HTTPS endpoint with Auth0 JWT validation at the
application layer for the development environment. Defer Cloud Load Balancing + Cloud
Armor to the production-readiness ADR.

**Why:** Cloud Run provides a managed HTTPS endpoint with Google-managed TLS by default.
For the dev environment, application-level JWT validation (rejecting unauthenticated
requests in FastAPI middleware) is sufficient. Rate limiting, WAF rules, and DDoS
protection via Cloud Armor are not necessary at this stage but should be evaluated
before production launch. When the team adds Cloud Armor, it will be attached to a
global HTTP(S) load balancer in front of Cloud Run, which also enables custom domain
mapping and Cloud CDN for API responses if needed.

### 11. Environment Strategy

**Decision:** Separate GCP projects per environment, managed by Terraform workspaces or
separate variable files.

**Environments:**

- **dev** — active development; Cloud SQL may be stopped overnight to save cost; min
  instances = 0 everywhere
- **staging** — pre-production validation; mirrors production config at smaller scale;
  used for integration testing and demo
- **prod** — production; sizing, HA, and connection pooling to be defined in the
  production-readiness ADR

Each environment gets its own GCP project (e.g., `jabs-hub-dev`, `jabs-hub-staging`,
`jabs-hub-prod`), its own Cloud SQL instance, its own GCS buckets, and its own Cloud Run
services. Terraform state is stored in a dedicated GCS bucket (
`jabs-hub-terraform-state`) with state locking via Cloud Storage's generation-based
locking. Environment-specific configuration is managed through Terraform `.tfvars`
files.

---

## Alternatives Considered

### Alternative A: Single VM Deployment

Deploy all components (FastAPI, Angular via nginx, PostgreSQL, MinIO for storage) on a
single GCE virtual machine.

| Aspect                 | Assessment                                                                                                                 |
|------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **Simplicity**         | Single machine to manage; familiar mental model                                                                            |
| **Cost (idle)**        | Fixed monthly cost (~$25–50/mo for an e2-medium) regardless of usage                                                       |
| **Cost (active)**      | Cheaper than managed services at very low scale, but no scale-to-zero                                                      |
| **Operational burden** | Team must manage OS patching, PostgreSQL upgrades, backups, SSL certs, monitoring, disk management, and security hardening |
| **Scaling**            | Vertical only; single point of failure; no autoscaling                                                                     |
| **Disaster recovery**  | Manual; requires snapshot-based backup strategy                                                                            |
| **Deployment**         | SSH + systemd or Docker Compose; no built-in blue/green deploys                                                            |

**Why rejected:** The operational burden is too high for a 2–3 person team. The lack of
autoscaling, managed backups, and built-in HA makes this fragile for a platform intended
to grow. While cheaper at minimal scale, the team's time spent on sysadmin tasks
outweighs the savings. This approach also makes it harder to evolve toward multi-tenant
or multi-region deployment later.

### Alternative B: GKE (Google Kubernetes Engine)

Deploy FastAPI and the Angular frontend as Kubernetes workloads on a GKE cluster, with
Cloud SQL and GCS for data.

| Aspect                 | Assessment                                                                                                          |
|------------------------|---------------------------------------------------------------------------------------------------------------------|
| **Flexibility**        | Full control over networking, scheduling, sidecars, service mesh                                                    |
| **Cost**               | GKE Autopilot starts at ~$70–100/mo for cluster management + node costs; Standard mode requires managing node pools |
| **Operational burden** | Kubernetes expertise required: YAML manifests, Helm charts, ingress controllers, pod security, cluster upgrades     |
| **Scaling**            | Excellent horizontal scaling with HPA and cluster autoscaler                                                        |
| **Deployment**         | Sophisticated (Helm, ArgoCD, etc.) but complex                                                                      |

**Why rejected:** GKE is over-engineered for the current scale. The JABS Hub MVP serves
a small user base within the Kumar Lab and does not require the scheduling, networking,
or scaling capabilities of Kubernetes. The team would spend disproportionate time on
cluster operations (node pool sizing, ingress configuration, Helm chart maintenance)
relative to the application complexity. Cloud Run provides the container-based
deployment model without the Kubernetes operational overhead. If JABS Hub grows to
require multi-service orchestration, sidecar patterns, or complex networking, migration
from Cloud Run to GKE is well-documented and supported by GCP.

---

## Cost Estimates

The following are rough monthly estimates for the JABS Hub **development environment**
in a US region, assuming a small number of developers (2–5 active users) and initial
data seeding. All prices are approximate and based on GCP's published pricing as of
March 2026.

| Service                      | Configuration                                       | Estimated Monthly Cost                    |
|------------------------------|-----------------------------------------------------|-------------------------------------------|
| **Cloud Run (FastAPI)**      | Low traffic, scales to zero, ~100K requests/mo      | $0–15 (likely within free tier initially) |
| **Cloud Storage (frontend)** | Static Angular build + Cloud CDN, low bandwidth     | $1–5 (minimal storage + CDN serving)      |
| **Cloud Storage (data)**     | 500 GB Standard (growing), moderate egress          | $12–20 (storage) + $5–15 (egress)         |
| **Cloud CDN**                | Low request volume during dev                       | $1–3                                      |
| **HTTP(S) Load Balancer**    | Frontend LB for CDN + SSL                           | $18–20 (base forwarding rule cost)        |
| **Cloud SQL (PostgreSQL)**   | `db-custom-1-3840`, 10 GB SSD, single zone          | $30–50                                    |
| **Artifact Registry**        | Container images, <5 GB                             | $0–2                                      |
| **Auth0**                    | Free tier (up to 7,500 active users)                | $0                                        |
| **Secret Manager**           | <10 secrets, low access volume                      | $0 (free tier)                            |
| **Networking / Misc**        | VPC connector, DNS, logging, Terraform state bucket | $7–12                                     |
| **Total (dev environment)**  |                                                     | **~$75–140/month**                        |

**Scaling notes:**

- The HTTP(S) load balancer forwarding rule is the most notable fixed cost that was not
  present in a Firebase Hosting approach. This is the trade-off for full Terraform
  control and GCP-native CDN.
- Cloud SQL is the next largest fixed cost. The dev instance can be stopped during
  non-business hours via a scheduled Cloud Scheduler job to save ~50%.
- GCS costs scale linearly with data volume. At 5 TB of stored video, expect ~$
  100–115/mo for storage alone.
- Cloud Run costs remain low until traffic is sustained; the free tier includes 2
  million requests/month.
- For production with HA (Cloud SQL regional), expect the database cost to roughly
  double. Production cost estimates will be detailed in the production-readiness ADR.

---

## Consequences

### Positive

- **Low operational overhead.** Managed services (Cloud Run, Cloud SQL, GCS, Cloud CDN)
  eliminate the need for server patching, database administration, and manual scaling.
- **Cost-efficient at low scale.** Cloud Run's scale-to-zero keeps compute costs minimal
  during development and early adoption.
- **Clean separation of concerns.** Static frontend (GCS + CDN), API (Cloud Run),
  database (Cloud SQL), and data storage (GCS) are independently deployable and
  scalable.
- **Fully Terraform-managed.** All infrastructure — including the frontend hosting, load
  balancer, and CDN — is defined in Terraform and versioned alongside application code.
  No out-of-band Firebase project to manage.
- **Portable application layer.** FastAPI in a container can run on any container
  platform. The thin storage abstraction layer keeps the door open for backend
  migration.
- **Aligned with team experience.** The team has prior GCP + Cloud Run + FastAPI
  experience, reducing ramp-up time.
- **Extensible.** The architecture supports future growth: adding Cloud Run services,
  enabling Cloud SQL HA, introducing Pub/Sub for async processing, or migrating to GKE
  if orchestration needs grow.

### Negative

- **GCP vendor coupling.** While the application layer is portable, the infrastructure (
  Terraform modules, IAM bindings, GCS-specific signed URLs) is GCP-specific. Migration
  to AWS or Azure would require IaC rewriting and storage integration changes. The thin
  storage abstraction and MinIO-based testing mitigate this partially.
- **Cold starts.** Cloud Run with min instances set to 0 will experience cold starts (
  typically 1–3 seconds for a FastAPI container). For the JABS GUI, which makes API
  calls during interactive sessions, this may cause occasional latency spikes.
  Mitigation: set min instances to 1 for production if cold starts become disruptive (~$
  15–25/mo additional).
- **Cloud SQL always-on cost.** Unlike Cloud Run, Cloud SQL instances incur cost
  continuously. This is the largest fixed expense. The team should monitor utilization
  and right-size as needed.
- **Load balancer base cost.** The global HTTP(S) load balancer required for Cloud CDN +
  managed SSL has a fixed forwarding rule cost (~$18/mo) that exceeds what Firebase
  Hosting's free tier would cost. This is the trade-off for full Terraform control and
  avoiding a Firebase project dependency.
- **CDN cache invalidation on deploy.** Unlike Firebase Hosting's atomic deploys,
  deploying to GCS + Cloud CDN requires an explicit cache invalidation step for
  `index.html` after each frontend deploy. This is a minor operational step handled by
  the CI/CD pipeline.

### Risks and Mitigations

| Risk                                             | Mitigation                                                                                                                                                                                                  |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GCS egress costs grow with video download volume | Monitor egress closely; Cloud CDN caching reduces origin fetches; implement client-side caching in the JABS GUI                                                                                             |
| Cloud SQL undersized for production workload     | Dev sizing is intentionally minimal; production-readiness ADR will benchmark and right-size based on observed query patterns. Vertical scaling is straightforward (change machine type with brief downtime) |
| Cloud Run → Cloud SQL connection exhaustion      | SQLAlchemy pool limits documented; max instances × pool size kept within Cloud SQL connection limit; PgBouncer sidecar planned for production                                                               |
| Auth0 free tier limits exceeded                  | Auth0's free tier supports 7,500 active users, far beyond the expected user base; monitor usage                                                                                                             |
| Team unfamiliar with Terraform                   | Invest in initial Terraform onboarding; start with a simple module structure; use `terraform plan` in PR checks                                                                                             |
| Vendor lock-in limits future portability         | Thin storage abstraction, MinIO for local dev, containerized API all reduce switching cost                                                                                                                  |
| Accidental bulk deletion of GCS video data       | GCS object versioning provides recovery for individual object overwrites/deletes; for catastrophic deletion, evaluate Object Lock or a cross-region backup bucket before production launch                  |

---

## Related Decisions

- **ADR-000X (future):** Production readiness — instance sizing, HA, connection pooling,
  disaster recovery (RTO/RPO), Cloud Armor / WAF, backup validation strategy
- **ADR-000X (future):** Label storage format and versioning strategy
- **ADR-000X (future):** Envision integration model (reference vs. ingest vs. hybrid)
- **ADR-000X (future):** GA4GH DRS evaluation for cross-institutional data sharing
- **ADR-000X (future):** Auth0 desktop GUI authentication flow (device code vs. PKCE with
  browser redirect)

---

## References

- [GCP Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [GCP Cloud SQL Pricing](https://cloud.google.com/sql/pricing)
- [GCP Cloud Storage Pricing](https://cloud.google.com/storage/pricing)
- [Cloud CDN Documentation](https://cloud.google.com/cdn/docs)
- [Google Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Auth0 Documentation](https://auth0.com/docs)
