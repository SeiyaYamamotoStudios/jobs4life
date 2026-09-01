# Mireille Fontaine

## Anchorwell Payments

### Staff Backend Engineer (Jul 2023 - Present)

- Owns the settlement reconciliation service that closes each business day's payment batches across three currencies.
- Rebuilt the reconciliation pipeline from scratch after a 2023 data-loss incident during a database migration; it has processed every batch without a repeat incident since.
- Led migration of the payments ledger from a self-hosted Postgres cluster to a managed, multi-region database service.
- Worked on the fraud-signal ingestion path that feeds the risk team's models.
- Introduced a canary deployment process for the payments API.
- Participates in the org-wide incident response rotation for payment-critical services.

### Senior Backend Engineer (Feb 2021 - Jul 2023)

- Built the idempotency-key service used by the checkout team to prevent duplicate charges.
- Reduced peak-hour checkout latency by re-architecting the payment authorization path.
- Mentored two mid-level engineers joining the payments team.
- Wrote the internal runbook for PCI-DSS scope reduction that the security team adopted.
- Was the on-call primary for the payments service for six consecutive quarters.

## Harborline Payments

### Backend Engineer (Mar 2018 - Feb 2021)

- Implemented the retry and backoff logic for failed card-network calls.
- Worked on the merchant onboarding API alongside the platform team.
- Migrated batch settlement jobs from cron scripts to an Airflow-based pipeline.
- Investigated and fixed a rounding error in currency conversion that had gone unnoticed for several months.
- Contributed test coverage for the refund processing service.
- Presented the settlement pipeline redesign at the company's quarterly engineering demo.

## Voss Analytics

### Software Engineer (Aug 2015 - Feb 2018)

- Built ETL jobs that loaded client usage data into a reporting warehouse nightly.
- Wrote the first version of the internal API used by the analytics dashboard.
- Worked on performance tuning for slow SQL reports used by the customer success team.
- Helped set up the team's first CI pipeline using Jenkins.
- Provided on-call coverage for the reporting service during quarterly close.

## Kestrel Digital

### Junior Developer (Jul 2014 - Jul 2015)

- Built small internal tools in Python for the operations team.
- Fixed bugs reported against the client-facing booking website.
- Wrote documentation for the internal deployment process.
- Shadowed senior engineers during production releases.

## Things Stated Explicitly as NOT True, or Boundaries to Hold

- Has not held a formal people-management role; mentoring at Anchorwell was informal, with no budget or hiring authority attached.
- Has not worked in machine learning or computer-vision systems in any capacity.
- Has never held on-call responsibility for a service outside the payments domain.
- Has not built or operated a Kafka cluster; all messaging at Harborline and Anchorwell used a managed queue service, not self-run Kafka.
- Has not obtained a PCI-DSS QSA certification; the runbook work was engineering, not formal compliance auditing.
- Has no experience with Rust; all production work has been in Go, Python, and Java.
- Has not worked at a company with more than 400 engineers; all four employers were under 200 engineering staff.
- Has never presented at an external conference; the quarterly demo at Harborline was internal only.
- Has not directly owned a budget of any size.
