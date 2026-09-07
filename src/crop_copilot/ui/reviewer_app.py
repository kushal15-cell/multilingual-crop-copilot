from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Agronomist Review Queue", page_icon="🧑🏽‍🌾", layout="wide")
st.title("🧑🏽‍🌾 Agronomist Review Queue")
st.warning("Only a qualified, authorized reviewer should approve advice.")

reviewer_key = st.text_input("Reviewer API key", type="password")
reviewer_name = st.text_input("Reviewer name / license or staff ID")

if reviewer_key:
    response = httpx.get(
        f"{API_BASE}/v1/reviewer/queue",
        headers={"X-Reviewer-Key": reviewer_key},
        timeout=30,
    )
    if not response.is_success:
        st.error(response.json().get("detail", response.text))
    else:
        queue = response.json()
        st.metric("Pending cases", len(queue))
        for record in queue:
            with st.expander(
                f"{record['crop']} · {record['disease']} · {record['confidence']:.0%}",
                expanded=len(queue) == 1,
            ):
                st.write("Farmer question:", record["question"])
                st.write("Risk reasons:", record["risk"]["reasons"])
                st.subheader("Evidence")
                for evidence in record["evidence"]:
                    st.write(
                        f"**{evidence['source_id']}** — {evidence['title']} "
                        f"(expert validated: {evidence['expert_validated']})"
                    )
                    st.caption(evidence["text"])
                draft = record["private_draft"]
                summary = st.text_area("Summary", draft["summary"], key=f"summary_{record['id']}")
                actions_text = st.text_area(
                    "Actions (one per line)",
                    "\n".join(draft["actions"]),
                    key=f"actions_{record['id']}",
                )
                warnings_text = st.text_area(
                    "Warning signs (one per line)",
                    "\n".join(draft["warning_signs"]),
                    key=f"warnings_{record['id']}",
                )
                limitations = st.text_area(
                    "Limitations", draft["limitations"], key=f"limits_{record['id']}"
                )
                notes = st.text_area("Reviewer notes", key=f"notes_{record['id']}")
                approve_col, reject_col = st.columns(2)
                decision_url = f"{API_BASE}/v1/reviewer/{record['id']}/decision"
                headers = {"X-Reviewer-Key": reviewer_key}
                with approve_col:
                    if st.button("Approve edited response", key=f"approve_{record['id']}"):
                        payload = {
                            "action": "approve",
                            "reviewer": reviewer_name,
                            "notes": notes,
                            "approved_text": {
                                "summary": summary,
                                "actions": [
                                    x.strip() for x in actions_text.splitlines() if x.strip()
                                ],
                                "warning_signs": [
                                    x.strip() for x in warnings_text.splitlines() if x.strip()
                                ],
                                "limitations": limitations,
                                "sources": draft["sources"],
                            },
                        }
                        result = httpx.post(decision_url, headers=headers, json=payload, timeout=30)
                        if result.is_success:
                            st.success("Approved")
                            st.rerun()
                        else:
                            st.error(result.json().get("detail", result.text))
                with reject_col:
                    if st.button("Reject", key=f"reject_{record['id']}"):
                        payload = {
                            "action": "reject",
                            "reviewer": reviewer_name,
                            "notes": notes,
                        }
                        result = httpx.post(decision_url, headers=headers, json=payload, timeout=30)
                        if result.is_success:
                            st.success("Rejected")
                            st.rerun()
                        else:
                            st.error(result.json().get("detail", result.text))
