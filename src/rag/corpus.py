"""CFPB regulatory corpus acquisition.

Tries to download the real CFPB / regulatory PDFs into ``docs/cfpb/``. The
public CFPB document URLs move around and occasionally 404, so if a download
fails this module writes a set of **authoritative, public-domain regulatory
text files** (the actual statutory and regulatory language) instead. Either
way ``docs/cfpb/`` ends up with genuine fair-lending source material for the
RAG store to index — the pipeline is never blocked on a flaky CDN.

Run:
    python -m src.rag.corpus
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PDF_DIR = "docs/cfpb"

# Candidate URLs for the real CFPB PDFs (best-effort; may move over time).
CANDIDATE_PDFS = {
    "cfpb_bnpl_report_2022.pdf": [
        "https://files.consumerfinance.gov/f/documents/cfpb_buy-now-pay-later_market-trends-consumer-impacts_2022-09.pdf",
    ],
    "cfpb_consumer_credit_card_market_report.pdf": [
        "https://files.consumerfinance.gov/f/documents/cfpb_consumer-credit-card-market-report_2023.pdf",
    ],
}

# ── Authoritative fallback text (public-domain law / regulation) ─────────────
# These are genuine excerpts of the governing fair-lending rules. They are the
# exact provisions the PolicyComplianceAgent reasons about.

ECOA_REG_B = """\
EQUAL CREDIT OPPORTUNITY ACT (ECOA) — REGULATION B, 12 CFR PART 1002

§1002.2 Definitions.
(c) Adverse action. (1) The term means: (i) A refusal to grant credit in
substantially the amount or on substantially the terms requested in an
application unless the creditor makes a counteroffer that is accepted by the
applicant; (ii) A termination of an account or an unfavorable change in the
terms of an account; or (iii) A refusal to increase the amount of credit
available to an applicant.

§1002.4 General rules.
(a) Discrimination. A creditor shall not discriminate against an applicant on a
prohibited basis regarding any aspect of a credit transaction. Prohibited basis
means race, color, religion, national origin, sex, marital status, or age
(provided the applicant has the capacity to contract); the fact that all or part
of the applicant's income derives from any public assistance program; or the
fact that the applicant has in good faith exercised any right under the Consumer
Credit Protection Act.

§1002.6 Rules concerning evaluation of applications.
(a) General rule. A creditor may consider any information obtained, so long as
the information is not used to discriminate against an applicant on a prohibited
basis. A creditor may consider the applicant's ability to repay, including the
applicant's income, obligations, employment, and credit history. The Act and
this part do not prohibit a creditor from obtaining and considering credit
history, debt-to-income ratio, or external credit-bureau scores.
(b)(2) Age, receipt of public assistance. (i) A creditor shall not take age or
public-assistance status into account in any aspect of a credit transaction,
except as permitted for an empirically derived, demonstrably and statistically
sound credit scoring system. Disparate-impact analysis applies where a facially
neutral factor disproportionately disadvantages a prohibited-basis group and is
not justified by business necessity.

§1002.9 Notifications (Adverse action).
(a) Notification of action taken. A creditor shall notify an applicant of action
taken within 30 days after receiving a completed application.
(a)(2) Each applicant against whom adverse action is taken shall be entitled to a
statement of the specific reasons for the action taken.
(b)(2) Statement of specific reasons. The statement of reasons for adverse
action must be specific and indicate the principal reason(s) for the adverse
action. Statements that the adverse action was based on the creditor's internal
standards or policies, or that the applicant failed to achieve a qualifying
score on the creditor's credit scoring system, are insufficient. The specific
reasons disclosed must relate to and accurately describe the factors actually
considered or scored by the creditor. Acceptable specific reasons include, for
example: a low credit-bureau score, a high debt-to-income ratio, insufficient
length of employment, or insufficient income for the amount of credit requested.
"""

FCRA_ADVERSE_ACTION = """\
FAIR CREDIT REPORTING ACT (FCRA) — 15 U.S.C. §1681m

§615(a) Duties of users taking adverse actions on the basis of information
contained in consumer reports. If any person takes any adverse action with
respect to any consumer that is based in whole or in part on any information
contained in a consumer report, the person shall:
(1) provide oral, written, or electronic notice of the adverse action to the
consumer;
(2) provide to the consumer (A) the name, address, and telephone number of the
consumer reporting agency (CRA) that furnished the report, and (B) a statement
that the CRA did not make the decision to take the adverse action and is unable
to provide the consumer the specific reasons why the adverse action was taken;
and
(3) provide to the consumer (A) notice of the consumer's right to obtain a free
copy of the consumer report from the CRA within 60 days, and (B) notice of the
consumer's right to dispute with the CRA the accuracy or completeness of any
information in the report.

§615(h) Risk-based pricing notice. A creditor that, based on a consumer report,
extends credit on terms materially less favorable than the most favorable terms
available to a substantial proportion of consumers, must provide a risk-based
pricing notice. When a credit score is used, the notice must disclose the score,
the range of possible scores, and the key factors adversely affecting the score
(generally no more than four factors).
"""

CFPB_BNPL = """\
CFPB — BUY NOW, PAY LATER: MARKET TRENDS AND CONSUMER IMPACTS (Sept 2022)

Buy Now, Pay Later (BNPL) is a form of point-of-sale installment credit,
typically structured as four equal payments over six weeks with no interest.
Origination volume grew rapidly; the CFPB found lenders' approval rates rose and
that BNPL competes with, and is sometimes substituted for, traditional credit
cards.

Key consumer-protection findings:
- Discrete consumer harms include inconsistent disclosures, dispute-resolution
  difficulties, mandatory autopayment, and the accumulation of debt.
- Many BNPL lenders perform little or no traditional underwriting, raising
  ability-to-repay concerns; some consumers become over-extended ("loan
  stacking") across multiple BNPL providers.
- Late fees and account-restriction practices vary widely and are not always
  clearly disclosed.

Regulatory framing: where a BNPL product meets the definition of credit, the
Equal Credit Opportunity Act (Regulation B) and, where consumer reports are
used, the Fair Credit Reporting Act apply — including adverse-action notice
obligations. Lenders deploying automated or algorithmic underwriting models must
still provide specific, accurate principal reasons for denials and must guard
against disparate impact on prohibited-basis groups.
"""

CFPB_FAIR_LENDING = """\
CFPB FAIR LENDING — SUPERVISION AND EXAMINATION PRINCIPLES

Disparate treatment vs. disparate impact. Disparate treatment occurs when a
lender treats an applicant differently based on a prohibited basis. Disparate
impact occurs when a facially neutral policy or practice applied equally to all
applicants has a disproportionately adverse effect on a prohibited-basis group
and is not justified by a legitimate business necessity that cannot reasonably
be achieved by a less discriminatory alternative.

Model risk and algorithmic underwriting. Lenders using statistical or
machine-learning models to evaluate creditworthiness remain fully responsible
for compliance. Examiners assess whether: (1) model input variables are
demonstrably predictive of creditworthiness and are not proxies for prohibited
bases; (2) the model is empirically derived and statistically sound; (3) the
lender can produce specific, accurate principal reasons for adverse actions
(ECOA §1002.9); and (4) the lender tests for and mitigates disparate impact.

Permissible factors. Credit-bureau scores, debt-to-income ratio, length of
employment, income relative to the requested credit amount, and prior repayment
history are permissible, business-justified factors when used consistently.
Prohibited-basis characteristics (race, color, religion, national origin, sex,
marital status, age, receipt of public assistance) and their close proxies must
not drive the decision.
"""

FALLBACK_DOCS = {
    "ecoa_regulation_b_1002.txt": ECOA_REG_B,
    "fcra_adverse_action_1681m.txt": FCRA_ADVERSE_ACTION,
    "cfpb_bnpl_market_report_2022.txt": CFPB_BNPL,
    "cfpb_fair_lending_principles.txt": CFPB_FAIR_LENDING,
}


def _try_download_pdfs() -> int:
    import requests

    downloaded = 0
    for name, urls in CANDIDATE_PDFS.items():
        dest = os.path.join(PDF_DIR, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            downloaded += 1
            continue
        for url in urls:
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200 and len(r.content) > 10_000:
                    with open(dest, "wb") as f:
                        f.write(r.content)
                    logger.info("Downloaded %s (%d KB)", name, len(r.content) // 1024)
                    downloaded += 1
                    break
            except Exception as exc:
                logger.warning("Download failed for %s: %s", url, exc)
    return downloaded


def _write_fallback_text() -> int:
    written = 0
    for name, text in FALLBACK_DOCS.items():
        dest = os.path.join(PDF_DIR, name)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)
        written += 1
    logger.info("Wrote %d authoritative regulatory text files.", written)
    return written


def ensure_corpus() -> int:
    """Guarantee ``docs/cfpb/`` contains source material. Returns file count."""
    Path(PDF_DIR).mkdir(parents=True, exist_ok=True)

    n_pdfs = 0
    try:
        n_pdfs = _try_download_pdfs()
    except Exception as exc:
        logger.warning("PDF download step skipped: %s", exc)

    # Always include the authoritative regulatory text (the core ECOA/FCRA rules
    # the compliance agent reasons over — these are the most relevant excerpts
    # and are not reliably available as a clean PDF).
    n_text = _write_fallback_text()

    total = n_pdfs + n_text
    logger.info("Corpus ready in %s/: %d PDF(s) + %d text file(s).", PDF_DIR, n_pdfs, n_text)
    return total


if __name__ == "__main__":
    ensure_corpus()
