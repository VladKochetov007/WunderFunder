"""Conditionally stop only the accepted first-cell score unit after exit."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import selectors
import stat
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

SCORE_UNIT = "computer-vision-homework-fg19-first-cell-score-v3.service"
SCORE_INVOCATION = '9bd64d6759e8462bb40985b06a9a0ad6'
ACCEPTANCE_BYTES = 21229
ACCEPTANCE_SHA256 = '6e9a8162321b2490d5131039e4fe9cf9aded13e58b789d5789491558861c0ed8'
ACCEPTANCE_BASE64 = 'ewogICJzY2hlbWEiOiAiZmcxOS5maXJzdC1jZWxsLXNjb3JlLXJvb3QtYWNjZXB0YW5jZS52MSIsCiAgInN0YXR1cyI6ICJBQ0NFUFRFRF9GSVhFRF9GSVJTVF9DRUxMX0RFVkVMT1BNRU5UX1NDT1JFX1JFUE9SVCIsCiAgImNyZWF0ZWRfdXRjIjogIjIwMjYtMTAtMjFUMTM6MjM6MDYuNzI0MzgzKzAwOjAwIiwKICAidW5pdCI6ICJjb21wdXRlci12aXNpb24taG9tZXdvcmstZmcxOS1maXJzdC1jZWxsLXNjb3JlLXYzLnNlcnZpY2UiLAogICJpbnZvY2F0aW9uX2lkIjogIjliZDY0ZDY3NTllODQ2MmJiNDA5ODViMDZhOWEwYWQ2IiwKICAibmF0aXZlX3Rlcm1pbmFsIjogewogICAgImFjdGl2ZV9zdGF0ZSI6ICJhY3RpdmUiLAogICAgInN1Yl9zdGF0ZSI6ICJleGl0ZWQiLAogICAgIm1haW5fcGlkIjogMCwKICAgICJleGVjX21haW5fY29kZSI6IDEsCiAgICAiZXhlY19tYWluX3N0YXR1cyI6IDAKICB9LAogICJyZXN1bHQiOiB7CiAgICAicGF0aCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLTIwMjYxMDE4L3Jlc2VhcmNoX3RtcC9mZzE5X2ZpcnN0X2NlbGxfZjBfczEwMDFfZXZhbHVhdGlvbl92My9SRVNVTFQuanNvbiIsCiAgICAiYnl0ZXMiOiAzMzg1MiwKICAgICJzaGEyNTYiOiAiMTY0NmQ4NjU2YTA0MmNlNDliMzU1ZjIyNTIxMTM0MjA2ZDYwMWMxZGJhYzM2OTc5OTFmY2QzMDhhYTFhOTRkNSIsCiAgICAic2NoZW1hIjogImZnMTkuZmlyc3QtY2VsbC1maXZlLWFybS1zdHJlYW1pbmctc2NvcmUudjEiLAogICAgInN0YXR1cyI6ICJGSVhFRF9GSVJTVF9DRUxMX0RFVkVMT1BNRU5UX0VWSURFTkNFX1NDT1JFRCIKICB9LAogICJzY29yZV9yZWNlaXB0cyI6IHsKICAgICJhZG1pc3Npb24iOiB7CiAgICAgICJieXRlcyI6IDY5NjUwLAogICAgICAic2hhMjU2IjogImRkYjg4YThjNDU4NGFkYmI3NmQ2MjIwMjg1Y2YwNTBhODAwMzI0ZjU5ODRlZWE0OWYwZTdjNzQ4YTM1MTYyZDciCiAgICB9LAogICAgImxhdW5jaCI6IHsKICAgICAgImJ5dGVzIjogNDE5MywKICAgICAgInNoYTI1NiI6ICI4NDhiMGM2ZjA1NWEwOTNiMjYwMGMzMDQzY2I0MTIwNGQ2MDQ3NTY1ZTc5ZTE5OWJjMzY3YzE0Zjk0ZjRkMmU0IgogICAgfSwKICAgICJleHBvcnRfYWNjZXB0YW5jZSI6IHsKICAgICAgImJ5dGVzIjogODgzMywKICAgICAgInNoYTI1NiI6ICI1YTlhOTkxMjFmODBlYjdmYTA3ZTQ1OTEzMThiZWE4MTA3OGNhOGIyZGIxMDMyZjE1MGNjMzNhZmNjNzI4ZDBkIgogICAgfSwKICAgICJleHBvcnRfZXhhY3Rfc3RvcCI6IHsKICAgICAgImJ5dGVzIjogMTE0MCwKICAgICAgInNoYTI1NiI6ICI2OTVlMDZjNjA0NTJlOTY5MzQ5YjY0MzU2ZGNkNzk1YmNiYjQ1ODkyMzI4MjYxMTM2ZjEwNWI1Zjg2NzNiNzJmIgogICAgfSwKICAgICJzdGF0dXNfY2FwdHVyZSI6IHsKICAgICAgImJ5dGVzIjogMTQ4Mzc5LAogICAgICAic2hhMjU2IjogImVhYzhkZTZjNmUxZGNjNDZlMTIzMzFkZWMyNDNiZmJlOWI0YWNmMTg1NTM3MjQ2ZWRmMGU3ODNiZWNhYzUzYjIiCiAgICB9LAogICAgImV4cG9ydF9tYW5pZmVzdCI6IHsKICAgICAgImJ5dGVzIjogMjMxMzAsCiAgICAgICJzaGEyNTYiOiAiNmI5M2NhYmYwZTBiNDQ1MGExMTgyMzY4MmRmZWJhOTM5NjA3NmVmYmM3YjRhMjAzMzY3M2ZjNDZiN2YwNGMyMyIKICAgIH0KICB9LAogICJzb3VyY2UiOiB7CiAgICAic291cmNlX2NvbW1pdCI6ICI2OGZmNjgyOTc3YjgyMmYzODM1MWFhYzBmNjIxMTQ2ZjVmMTNiNDQwIiwKICAgICJzb3VyY2VfbWFuaWZlc3Rfc2hhMjU2IjogImE0NGY4YTNmZDNmZGExZWZkZGE1NzhjODY5NDAzNmRmZmFmNzJjZTY5OTllNGU5MDk1ZjYyOTgzNjViMzU2MjEiLAogICAgInNvdXJjZV9zZXRfc2hhMjU2IjogIjljNDYxYmNjNTg4MzU3MmEwNzFkNDFhODZlMThiOWE5MTEzNjA5OWNmOGZhODczNzFjZTdjM2VlNDE0NDViZDgiLAogICAgInNwbGl0X21hbmlmZXN0X3NoYTI1NiI6ICJjMDlmNzI0MDM5NmEzM2NhODI5NzcxMTVhZjhkNjA2N2U1NzFjMGVjMjVjNThhM2M5OWI0MzhjYTJkMjI5NGI3IiwKICAgICJvcmRlcmVkX21lbWJlcnNoaXBfc2hhMjU2IjogIjE4ZjBjOTUxNGQyNWU5NGNiZTVlYjI0NjA3ODc5MzI0YmI4M2QxMTdkOTYyNjcwZDRkZGY5NzMzNjQ1ZjRmMGMiLAogICAgImV4cG9ydF9tYW5pZmVzdCI6IHsKICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9maXJzdF9jZWxsX2YwX3MxMDAxX2V2YWx1YXRpb25fdjMvZXhwb3J0cy9FWFBPUlRfTUFOSUZFU1RfVjEuanNvbiIsCiAgICAgICJieXRlcyI6IDIzMTMwLAogICAgICAic2hhMjU2IjogIjZiOTNjYWJmMGUwYjQ0NTBhMTE4MjM2ODJkZmViYTkzOTYwNzZlZmJjN2I0YTIwMzM2NzNmYzQ2YjdmMDRjMjMiCiAgICB9LAogICAgImZpeGVkX21vZGVscyI6IFsKICAgICAgewogICAgICAgICJhcm0iOiAiY29udHJvbCIsCiAgICAgICAgInJ1bl9uYW1lIjogImZnMTlfY2hpbGRfY29udHJvbF9mMF9zMTAwMV9maXRfdjEiLAogICAgICAgICJ1bml0IjogImNvbXB1dGVyLXZpc2lvbi1ob21ld29yay1mZzE5LWNoaWxkLWNvbnRyb2wtZml0LXYxLnNlcnZpY2UiLAogICAgICAgICJpbnZvY2F0aW9uX2lkIjogImU2NmUwZDA3MDMzMTRlNGZiNWQ0N2RjZGJmY2FmMjAyIiwKICAgICAgICAicmVwcmVzZW50YXRpb25fbW9kZSI6ICJiYXNlIiwKICAgICAgICAic2VsZWN0ZWRfZXBvY2giOiA1LAogICAgICAgICJzdWNjZXNzZnVsX3VwZGF0ZXMiOiA0NDc2MCwKICAgICAgICAiY2hlY2twb2ludCI6IHsKICAgICAgICAgICJieXRlcyI6IDQyNzk5NzAsCiAgICAgICAgICAicmVsYXRpdmVfcGF0aCI6ICJhcnRpZmFjdHMvcnVucy9mZzE5X2NoaWxkX2NvbnRyb2xfZjBfczEwMDFfZml0X3YxL21vZGVsX2VwNS5wdCIsCiAgICAgICAgICAic2hhMjU2IjogImQzZjI1ZTUyYWE1OTAwMjk1OTNlNWFmNmVhMDZjNmQ0ZjQ2MGJjNjEzMWZmYTg1ZDRjZjFjYjA5YzM2YjNiM2YiCiAgICAgICAgfSwKICAgICAgICAidGVybWluYWxfYWNjZXB0YW5jZSI6IHsKICAgICAgICAgICJhYnNvbHV0ZV9wYXRoIjogIi9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvcmVzZWFyY2hfdG1wL2ZnMTlfY2hpbGRfYWxsdGhyZWVfZml0X3YxL0NPTlRST0xfVEVSTUlOQUxfQUNDRVBUQU5DRS5qc29uIiwKICAgICAgICAgICJieXRlcyI6IDExODgzLAogICAgICAgICAgInJlbGF0aXZlX3BhdGgiOiAiQ09OVFJPTF9URVJNSU5BTF9BQ0NFUFRBTkNFLmpzb24iLAogICAgICAgICAgInJvb3Rfa2V5IjogImZpdF9yZWNlaXB0c19yb290IiwKICAgICAgICAgICJzaGEyNTYiOiAiOTJhYWY1ZWVlZDE1NmFhYWQyNWM0YWRiYjY3MDBkNDFiMDUyNzYwMGI1MjBkZDA1MTY1N2E0M2E1MGJiYjZkMiIKICAgICAgICB9LAogICAgICAgICJleGFjdF9zdG9wIjogewogICAgICAgICAgImFic29sdXRlX3BhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9jaGlsZF9hbGx0aHJlZV9maXRfdjEvQ09OVFJPTF9FWEFDVF9TVE9QLmpzb24iLAogICAgICAgICAgImJ5dGVzIjogMzA4OSwKICAgICAgICAgICJyZWxhdGl2ZV9wYXRoIjogIkNPTlRST0xfRVhBQ1RfU1RPUC5qc29uIiwKICAgICAgICAgICJyb290X2tleSI6ICJmaXRfcmVjZWlwdHNfcm9vdCIsCiAgICAgICAgICAic2hhMjU2IjogIjYwMGNiM2M0YTU2YzdkMDczMWVkMmJmNjE3NGQ0N2Y1YjA5NjIzYzdjYTIyYTdlNWZjZTc3ZDRkMzU4YWUwMTIiCiAgICAgICAgfQogICAgICB9LAogICAgICB7CiAgICAgICAgImFybSI6ICJlbWEiLAogICAgICAgICJydW5fbmFtZSI6ICJmZzE5X2NoaWxkX2VtYV9mMF9zMTAwMV9maXRfdjEiLAogICAgICAgICJ1bml0IjogImNvbXB1dGVyLXZpc2lvbi1ob21ld29yay1mZzE5LWNoaWxkLWVtYS1maXQtdjEuc2VydmljZSIsCiAgICAgICAgImludm9jYXRpb25faWQiOiAiZGE1YmFmODJkMjJiNGYwMjg5MGUxODQxOTNiOGNhZjciLAogICAgICAgICJyZXByZXNlbnRhdGlvbl9tb2RlIjogImVtYSIsCiAgICAgICAgInNlbGVjdGVkX2Vwb2NoIjogNSwKICAgICAgICAic3VjY2Vzc2Z1bF91cGRhdGVzIjogNDQ3NjAsCiAgICAgICAgImNoZWNrcG9pbnQiOiB7CiAgICAgICAgICAiYnl0ZXMiOiA0Mjc5OTcwLAogICAgICAgICAgInJlbGF0aXZlX3BhdGgiOiAiYXJ0aWZhY3RzL3J1bnMvZmcxOV9jaGlsZF9lbWFfZjBfczEwMDFfZml0X3YxL21vZGVsX2VwNS5wdCIsCiAgICAgICAgICAic2hhMjU2IjogIjZlOTBiYzRkYmE4YWRkMGNhZjY3MDA5NzUxOWYwOGYzY2UyMzlkMGI2NGMzN2M0N2IzNzYwYWEyMmFmNDE4NjIiCiAgICAgICAgfSwKICAgICAgICAidGVybWluYWxfYWNjZXB0YW5jZSI6IHsKICAgICAgICAgICJhYnNvbHV0ZV9wYXRoIjogIi9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvcmVzZWFyY2hfdG1wL2ZnMTlfY2hpbGRfYWxsdGhyZWVfZml0X3YxL0VNQV9URVJNSU5BTF9BQ0NFUFRBTkNFLmpzb24iLAogICAgICAgICAgImJ5dGVzIjogMTMwMTYsCiAgICAgICAgICAicmVsYXRpdmVfcGF0aCI6ICJFTUFfVEVSTUlOQUxfQUNDRVBUQU5DRS5qc29uIiwKICAgICAgICAgICJyb290X2tleSI6ICJmaXRfcmVjZWlwdHNfcm9vdCIsCiAgICAgICAgICAic2hhMjU2IjogIjBhZWE2Mjk1YjY0NjcwYmIwMmFjMDA4ZmRmZWNlNmVlNDMwYjhmNjhiODQ5MDRhMTRjMzlkZDgxNmMwMTYxYTgiCiAgICAgICAgfSwKICAgICAgICAiZXhhY3Rfc3RvcCI6IHsKICAgICAgICAgICJhYnNvbHV0ZV9wYXRoIjogIi9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvcmVzZWFyY2hfdG1wL2ZnMTlfY2hpbGRfYWxsdGhyZWVfZml0X3YxL0VNQV9FWEFDVF9TVE9QLmpzb24iLAogICAgICAgICAgImJ5dGVzIjogMzA5MywKICAgICAgICAgICJyZWxhdGl2ZV9wYXRoIjogIkVNQV9FWEFDVF9TVE9QLmpzb24iLAogICAgICAgICAgInJvb3Rfa2V5IjogImZpdF9yZWNlaXB0c19yb290IiwKICAgICAgICAgICJzaGEyNTYiOiAiNTQ4ZmU5ODc2OTQ3MzQ1ZDRhYzQ0MzYwMWU0NzgzNDFiZGZlMTY1ZDU2NzBkNzM5N2Q1MGI0MDFiOTNjYzJiZiIKICAgICAgICB9CiAgICAgIH0sCiAgICAgIHsKICAgICAgICAiYXJtIjogInNwYXJzZSIsCiAgICAgICAgInJ1bl9uYW1lIjogImZnMTlfY2hpbGRfZGljdGlvbmFyeV9mMF9zMTAwMV9maXRfdjEiLAogICAgICAgICJ1bml0IjogImNvbXB1dGVyLXZpc2lvbi1ob21ld29yay1mZzE5LWNoaWxkLWRpY3Rpb25hcnktZml0LXYxLnNlcnZpY2UiLAogICAgICAgICJpbnZvY2F0aW9uX2lkIjogIjczYjEwYjY5OWUwMzRhMWI5OTUyYjIzNzdiODQ1MDk1IiwKICAgICAgICAicmVwcmVzZW50YXRpb25fbW9kZSI6ICJzcGFyc2UiLAogICAgICAgICJzZWxlY3RlZF9lcG9jaCI6IDUsCiAgICAgICAgInN1Y2Nlc3NmdWxfdXBkYXRlcyI6IDQ0NzYwLAogICAgICAgICJjaGVja3BvaW50IjogewogICAgICAgICAgImJ5dGVzIjogNDI3OTk3MCwKICAgICAgICAgICJyZWxhdGl2ZV9wYXRoIjogImFydGlmYWN0cy9ydW5zL2ZnMTlfY2hpbGRfZGljdGlvbmFyeV9mMF9zMTAwMV9maXRfdjEvbW9kZWxfZXA1LnB0IiwKICAgICAgICAgICJzaGEyNTYiOiAiYjg0MGQ5YzgxZDQ2NDA1ZjJkMjQzOWI5NGFkNmVmMTVmZjg1MDFiYjE0NzE1ZWM3OTc5MmQ5ODBlZmJmNzE1MSIKICAgICAgICB9LAogICAgICAgICJ0ZXJtaW5hbF9hY2NlcHRhbmNlIjogewogICAgICAgICAgImFic29sdXRlX3BhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9jaGlsZF9hbGx0aHJlZV9maXRfdjEvRElDVElPTkFSWV9DSElMRF9URVJNSU5BTF9BQ0NFUFRBTkNFLmpzb24iLAogICAgICAgICAgImJ5dGVzIjogMTQ5MTQsCiAgICAgICAgICAicmVsYXRpdmVfcGF0aCI6ICJESUNUSU9OQVJZX0NISUxEX1RFUk1JTkFMX0FDQ0VQVEFOQ0UuanNvbiIsCiAgICAgICAgICAicm9vdF9rZXkiOiAiZml0X3JlY2VpcHRzX3Jvb3QiLAogICAgICAgICAgInNoYTI1NiI6ICIyODNjNDQ3Yjg2YTljYzUwYzJmZmFiODQ5YTBkMDgzOTMwN2FjY2FkZTEwODNiYTUzZGNjNzIxN2E5ZGE4MmIxIgogICAgICAgIH0sCiAgICAgICAgImV4YWN0X3N0b3AiOiB7CiAgICAgICAgICAiYWJzb2x1dGVfcGF0aCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLTIwMjYxMDE4L3Jlc2VhcmNoX3RtcC9mZzE5X2NoaWxkX2FsbHRocmVlX2ZpdF92MS9ESUNUSU9OQVJZX0NISUxEX0VYQUNUX1NUT1AuanNvbiIsCiAgICAgICAgICAiYnl0ZXMiOiAzMTEzLAogICAgICAgICAgInJlbGF0aXZlX3BhdGgiOiAiRElDVElPTkFSWV9DSElMRF9FWEFDVF9TVE9QLmpzb24iLAogICAgICAgICAgInJvb3Rfa2V5IjogImZpdF9yZWNlaXB0c19yb290IiwKICAgICAgICAgICJzaGEyNTYiOiAiZjFhNzNmYzFiZjEzOTQzZTFkY2M5Y2YwMThlYmQ0MjI1OTg2YzRjZmNkY2NlYTFiNzFjNWVlOTYxYTU0YTVhMyIKICAgICAgICB9CiAgICAgIH0sCiAgICAgIHsKICAgICAgICAiYXJtIjogIm5sbCIsCiAgICAgICAgInJ1bl9uYW1lIjogImZnMTlfY2hpbGRfbmxsX2YwX3MxMDAxX2ZpdF92MSIsCiAgICAgICAgInVuaXQiOiAiY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLWZnMTktY2hpbGQtbmxsLWZpdC12MS5zZXJ2aWNlIiwKICAgICAgICAiaW52b2NhdGlvbl9pZCI6ICJjODMzYzYxM2ZiOTI0MjY5ODFhYjI1NWE1ZmFmNDQ1MiIsCiAgICAgICAgInJlcHJlc2VudGF0aW9uX21vZGUiOiAiYmFzZSIsCiAgICAgICAgInNlbGVjdGVkX2Vwb2NoIjogNSwKICAgICAgICAic3VjY2Vzc2Z1bF91cGRhdGVzIjogNDQ3NjAsCiAgICAgICAgImNoZWNrcG9pbnQiOiB7CiAgICAgICAgICAiYnl0ZXMiOiA0Mjc5OTcwLAogICAgICAgICAgInJlbGF0aXZlX3BhdGgiOiAiYXJ0aWZhY3RzL3J1bnMvZmcxOV9jaGlsZF9ubGxfZjBfczEwMDFfZml0X3YxL21vZGVsX2VwNS5wdCIsCiAgICAgICAgICAic2hhMjU2IjogImIxOWE2MTM2M2I0YjQ5Y2Q5OGQwMjc3ZjJmNzRmZDNhYTNkYjUwYmI2YzFmY2IzMTRiNWZhYjU3MWUyN2ZkNTgiCiAgICAgICAgfSwKICAgICAgICAidGVybWluYWxfYWNjZXB0YW5jZSI6IHsKICAgICAgICAgICJhYnNvbHV0ZV9wYXRoIjogIi9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvcmVzZWFyY2hfdG1wL2ZnMTlfY2hpbGRfYWxsdGhyZWVfZml0X3YxL05MTF9DSElMRF9URVJNSU5BTF9BQ0NFUFRBTkNFLmpzb24iLAogICAgICAgICAgImJ5dGVzIjogMTU4NjAsCiAgICAgICAgICAicmVsYXRpdmVfcGF0aCI6ICJOTExfQ0hJTERfVEVSTUlOQUxfQUNDRVBUQU5DRS5qc29uIiwKICAgICAgICAgICJyb290X2tleSI6ICJmaXRfcmVjZWlwdHNfcm9vdCIsCiAgICAgICAgICAic2hhMjU2IjogImQ5NDQ1N2IxMWNlMjJhMzY4Y2RjOTNkZjAyZTQ3M2M5YTgyY2RlNmRkZWJkMjZjYzY5NTdiYWZjMTQzNzUwYzUiCiAgICAgICAgfSwKICAgICAgICAiZXhhY3Rfc3RvcCI6IHsKICAgICAgICAgICJhYnNvbHV0ZV9wYXRoIjogIi9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvcmVzZWFyY2hfdG1wL2ZnMTlfY2hpbGRfYWxsdGhyZWVfZml0X3YxL05MTF9DSElMRF9FWEFDVF9TVE9QLmpzb24iLAogICAgICAgICAgImJ5dGVzIjogMzE0MCwKICAgICAgICAgICJyZWxhdGl2ZV9wYXRoIjogIk5MTF9DSElMRF9FWEFDVF9TVE9QLmpzb24iLAogICAgICAgICAgInJvb3Rfa2V5IjogImZpdF9yZWNlaXB0c19yb290IiwKICAgICAgICAgICJzaGEyNTYiOiAiYzYzYmU3OWIyN2QxMzFhY2IwYTM4ZDJhZDY0MWZhN2Q3MjM0NGNjMzQ0YzI2YWI5NGNhNzdmYWYzYjRhNzRmMSIKICAgICAgICB9CiAgICAgIH0sCiAgICAgIHsKICAgICAgICAiYXJtIjogImFsbHRocmVlIiwKICAgICAgICAicnVuX25hbWUiOiAiZmcxOV9jaGlsZF9hbGx0aHJlZV9mMF9zMTAwMV9maXRfdjEiLAogICAgICAgICJ1bml0IjogImNvbXB1dGVyLXZpc2lvbi1ob21ld29yay1mZzE5LWNoaWxkLWFsbHRocmVlLWZpdC12MS5zZXJ2aWNlIiwKICAgICAgICAiaW52b2NhdGlvbl9pZCI6ICI5NGQzYTcxMzJhNDM0ZDQ0ODM0NWIwMjA2YmQzZGFhMyIsCiAgICAgICAgInJlcHJlc2VudGF0aW9uX21vZGUiOiAiZW1hX3NwYXJzZSIsCiAgICAgICAgInNlbGVjdGVkX2Vwb2NoIjogNSwKICAgICAgICAic3VjY2Vzc2Z1bF91cGRhdGVzIjogNDQ3NjAsCiAgICAgICAgImNoZWNrcG9pbnQiOiB7CiAgICAgICAgICAiYnl0ZXMiOiA0Mjc5OTcwLAogICAgICAgICAgInJlbGF0aXZlX3BhdGgiOiAiYXJ0aWZhY3RzL3J1bnMvZmcxOV9jaGlsZF9hbGx0aHJlZV9mMF9zMTAwMV9maXRfdjEvbW9kZWxfZXA1LnB0IiwKICAgICAgICAgICJzaGEyNTYiOiAiNzA3MjA3NzA5OTc0NGU2OGZmMjNkZTNlMzkwYjE0NjkyN2MyM2RhMGUxYzU0ZjViYjE4ZWIyZmIwNDY3NmMwNyIKICAgICAgICB9LAogICAgICAgICJ0ZXJtaW5hbF9hY2NlcHRhbmNlIjogewogICAgICAgICAgImFic29sdXRlX3BhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9maXJzdF9jZWxsX2YwX3MxMDAxX2V2YWx1YXRpb25fdjMvYWNjZXB0ZWRfcmVjZWlwdHMvQUxMVEhSRUVfVEVSTUlOQUxfQUNDRVBUQU5DRS5qc29uIiwKICAgICAgICAgICJieXRlcyI6IDIwMjA4LAogICAgICAgICAgInJlbGF0aXZlX3BhdGgiOiAiYWNjZXB0ZWRfcmVjZWlwdHMvQUxMVEhSRUVfVEVSTUlOQUxfQUNDRVBUQU5DRS5qc29uIiwKICAgICAgICAgICJyb290X2tleSI6ICJldmlkZW5jZV9yb290IiwKICAgICAgICAgICJzaGEyNTYiOiAiZmFkYTUwMGVkZTYzYTE3NWQ3ZDA1NWQzOGZiZWY1ZDM4MTEwYWVmNDgxYTkxNzcxYWUzYzU3MTdjMWU3MDI2ZiIKICAgICAgICB9LAogICAgICAgICJleGFjdF9zdG9wIjogewogICAgICAgICAgImFic29sdXRlX3BhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9maXJzdF9jZWxsX2YwX3MxMDAxX2V2YWx1YXRpb25fdjMvYWNjZXB0ZWRfcmVjZWlwdHMvQUxMVEhSRUVfRVhBQ1RfU1RPUC5qc29uIiwKICAgICAgICAgICJieXRlcyI6IDMyMDgsCiAgICAgICAgICAicmVsYXRpdmVfcGF0aCI6ICJhY2NlcHRlZF9yZWNlaXB0cy9BTExUSFJFRV9FWEFDVF9TVE9QLmpzb24iLAogICAgICAgICAgInJvb3Rfa2V5IjogImV2aWRlbmNlX3Jvb3QiLAogICAgICAgICAgInNoYTI1NiI6ICIwMTU0YjNmMzVjNGZiODQ5ZDBkYjdlN2I2MGNlMWY0Y2UyZmEwMWFiMjIwM2MxN2RiNjU4NzdmODkzNTNmMTNkIgogICAgICAgIH0KICAgICAgfQogICAgXSwKICAgICJleHBvcnRlZF9maWxlX3BpbnMiOiB7CiAgICAgICJjb250cm9sIjogewogICAgICAgICJwcmVkaWN0aW9uIjogewogICAgICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9maXJzdF9jZWxsX2YwX3MxMDAxX2V2YWx1YXRpb25fdjMvZXhwb3J0cy9jb250cm9sL3ByZWRpY3Rpb25zLm5weSIsCiAgICAgICAgICAiYnl0ZXMiOiA3NDg4MDEyOCwKICAgICAgICAgICJzaGEyNTYiOiAiNmFmMmIwMTgzN2U3ODJmNGE4ZDMzNGVhOTA3ODllNjk0MzFiNmQwNTkwMzBhYTVjZGFiOTdhZmIwZmZiZjhjNiIKICAgICAgICB9LAogICAgICAgICJzaWRlY2FyIjogewogICAgICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9maXJzdF9jZWxsX2YwX3MxMDAxX2V2YWx1YXRpb25fdjMvZXhwb3J0cy9jb250cm9sL1BSRURJQ1RJT05fU0lERUNBUi5qc29uIiwKICAgICAgICAgICJieXRlcyI6IDk5NTgsCiAgICAgICAgICAic2hhMjU2IjogIjM3YTYzNjc0N2NjODFiOGI1NzQ0Y2I4NGM1ZDA4ZWMwNmNmOTMyNDNiNjk2MjI1OWUyZDllNzNjZTkwODgxNTUiCiAgICAgICAgfQogICAgICB9LAogICAgICAiZW1hIjogewogICAgICAgICJwcmVkaWN0aW9uIjogewogICAgICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9maXJzdF9jZWxsX2YwX3MxMDAxX2V2YWx1YXRpb25fdjMvZXhwb3J0cy9lbWEvcHJlZGljdGlvbnMubnB5IiwKICAgICAgICAgICJieXRlcyI6IDc0ODgwMTI4LAogICAgICAgICAgInNoYTI1NiI6ICJlZGNkMjIwMzgyMjcwZTc0YWNhOTkwNDVhOThhNDM1N2MxYThhMTEyYTk5YjYwYWQwYTI0YWM3ODZhNTZmY2U3IgogICAgICAgIH0sCiAgICAgICAgInNpZGVjYXIiOiB7CiAgICAgICAgICAicGF0aCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLTIwMjYxMDE4L3Jlc2VhcmNoX3RtcC9mZzE5X2ZpcnN0X2NlbGxfZjBfczEwMDFfZXZhbHVhdGlvbl92My9leHBvcnRzL2VtYS9QUkVESUNUSU9OX1NJREVDQVIuanNvbiIsCiAgICAgICAgICAiYnl0ZXMiOiA5OTI0LAogICAgICAgICAgInNoYTI1NiI6ICJmMGZjYmViZDhjNGFmYjFmM2I5ZGNmNmUyZTYxNDU3ODM5MjViMjI2ZTMxMGMyNWQ1NTgzNDgyNTg4NjcxNTA4IgogICAgICAgIH0KICAgICAgfSwKICAgICAgInNwYXJzZSI6IHsKICAgICAgICAicHJlZGljdGlvbiI6IHsKICAgICAgICAgICJwYXRoIjogIi9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvcmVzZWFyY2hfdG1wL2ZnMTlfZmlyc3RfY2VsbF9mMF9zMTAwMV9ldmFsdWF0aW9uX3YzL2V4cG9ydHMvc3BhcnNlL3ByZWRpY3Rpb25zLm5weSIsCiAgICAgICAgICAiYnl0ZXMiOiA3NDg4MDEyOCwKICAgICAgICAgICJzaGEyNTYiOiAiY2I1NmZjYWVhMjVlNjk1M2Q4Y2U4NDZjNGVhZGVlZjFjODdhYmEyYTY0M2U0ZGEyNWMwNDg3NzBkZTNiMjFhYiIKICAgICAgICB9LAogICAgICAgICJzaWRlY2FyIjogewogICAgICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9maXJzdF9jZWxsX2YwX3MxMDAxX2V2YWx1YXRpb25fdjMvZXhwb3J0cy9zcGFyc2UvUFJFRElDVElPTl9TSURFQ0FSLmpzb24iLAogICAgICAgICAgImJ5dGVzIjogMTAwMDYsCiAgICAgICAgICAic2hhMjU2IjogImRlYWQ1OWQ5N2NjOTU2ODY3MGNmOThhODI0ZmYyNjgwMmUxZmQwZTA1MzdhZTM5ZTg3MWRkNmJlZmNjZDUwOTQiCiAgICAgICAgfQogICAgICB9LAogICAgICAibmxsIjogewogICAgICAgICJwcmVkaWN0aW9uIjogewogICAgICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9maXJzdF9jZWxsX2YwX3MxMDAxX2V2YWx1YXRpb25fdjMvZXhwb3J0cy9ubGwvcHJlZGljdGlvbnMubnB5IiwKICAgICAgICAgICJieXRlcyI6IDc0ODgwMTI4LAogICAgICAgICAgInNoYTI1NiI6ICJjYWI2ZTViMmQyMjVjOTIzNmY3MmIwYmI3NDcwMzRiZTgyZmI1ZTNmNGIzMTljODY3ZDIwZjkxM2U3MzU3MGQ4IgogICAgICAgIH0sCiAgICAgICAgInNpZGVjYXIiOiB7CiAgICAgICAgICAicGF0aCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLTIwMjYxMDE4L3Jlc2VhcmNoX3RtcC9mZzE5X2ZpcnN0X2NlbGxfZjBfczEwMDFfZXZhbHVhdGlvbl92My9leHBvcnRzL25sbC9QUkVESUNUSU9OX1NJREVDQVIuanNvbiIsCiAgICAgICAgICAiYnl0ZXMiOiA5OTUwLAogICAgICAgICAgInNoYTI1NiI6ICJjZjI3MzEzNzI3MDQ5NDcxMjk3ZTdkYjc1MWMwYzYzZWRiNDU1ZmM2YjQ1MTZmNDJjZGM3MDIwYTExNDdhN2Q2IgogICAgICAgIH0KICAgICAgfSwKICAgICAgImFsbHRocmVlIjogewogICAgICAgICJwcmVkaWN0aW9uIjogewogICAgICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZXNlYXJjaF90bXAvZmcxOV9maXJzdF9jZWxsX2YwX3MxMDAxX2V2YWx1YXRpb25fdjMvZXhwb3J0cy9hbGx0aHJlZS9wcmVkaWN0aW9ucy5ucHkiLAogICAgICAgICAgImJ5dGVzIjogNzQ4ODAxMjgsCiAgICAgICAgICAic2hhMjU2IjogImFmYWVjOGRhOGI3NWI1Y2JlYzc2MDVhYjlkOGFlZjFlMzVkYWMzMmUxMmQ0NDRhM2NiMjAzMmI1NmY2Njg1NzUiCiAgICAgICAgfSwKICAgICAgICAic2lkZWNhciI6IHsKICAgICAgICAgICJwYXRoIjogIi9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvcmVzZWFyY2hfdG1wL2ZnMTlfZmlyc3RfY2VsbF9mMF9zMTAwMV9ldmFsdWF0aW9uX3YzL2V4cG9ydHMvYWxsdGhyZWUvUFJFRElDVElPTl9TSURFQ0FSLmpzb24iLAogICAgICAgICAgImJ5dGVzIjogMTAwNjYsCiAgICAgICAgICAic2hhMjU2IjogImUxYWUzOGZkZmM1ZjFlN2QxNjgwNTI4ZTg0MWM3YjhjNTE5ZDgyYWVmNDhkNGI2NjcyOTRhNjA1OTE4NmI0NzEiCiAgICAgICAgfQogICAgICB9CiAgICB9LAogICAgInJlZ2lzdGVyZWRfZGF0YSI6IHsKICAgICAgImFkbWlzc2lvbiI6IHsKICAgICAgICAiYWJzb2x1dGVfcGF0aCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLTIwMjYxMDE4L3Jlc2VhcmNoX3RtcC9mZzE5X2NoaWxkX2FsbHRocmVlX2ZpdF92MS9BRE1JU1NJT04uanNvbiIsCiAgICAgICAgImJ5dGVzIjogMTI5MTIsCiAgICAgICAgInJlbGF0aXZlX3BhdGgiOiAiQURNSVNTSU9OLmpzb24iLAogICAgICAgICJyb290X2tleSI6ICJmaXRfcmVjZWlwdHNfcm9vdCIsCiAgICAgICAgInNoYTI1NiI6ICI1MzEyYWU1ZmVmZmM2N2Q3YTMzMDUxMzc5YWZhYmI5YzEzNGRkOTQ2ZjFmOTczMTVkODM5YzdkNmYxMGQ1NTQwIgogICAgICB9LAogICAgICAicG9ydGFibGVfbWFuaWZlc3QiOiB7CiAgICAgICAgImFic29sdXRlX3BhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9zdXBwb3J0LXNvdXJjZS90ZXJtaW5hbC12MS9QT1JUQUJMRV9TT1VSQ0VTLmpzb24iLAogICAgICAgICJieXRlcyI6IDIxNTcsCiAgICAgICAgInNoYTI1NiI6ICIxZjM5Y2JmOGIyNDI0OTVlMThiYWFmNWQzNmVlYWM5MDdjMGFkNjU1ZTk1MWIwZWZhNmE0ODU4MGI3Njk5Mzc4IgogICAgICB9LAogICAgICAicmVnaXN0ZXJlZF9zb3VyY2VfaWRlbnRpdGllcyI6IHsKICAgICAgICAiY2FjaGUvY2hhaW5fZm9sZHMubnB6IjogewogICAgICAgICAgImJ5dGVzIjogNDYyNzE5LAogICAgICAgICAgImRldmljZSI6IDY0NTEyLAogICAgICAgICAgImlub2RlIjogNzM1NDgzNTQ5LAogICAgICAgICAgImlzbGluayI6IHRydWUsCiAgICAgICAgICAibXRpbWVfbnMiOiAxNzkyMjcyNDc2MDM0OTI4NTI0LAogICAgICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9kYXRhL2NhY2hlL2NoYWluX2ZvbGRzLm5weiIsCiAgICAgICAgICAicmVnaXN0ZXJlZF9zaGEyNTYiOiAiYTExMmY4MDkyNTlmNjMxYzJlZTdjMTI0ODBlNGVjMWQ0ZWVmMTk3YTk4MGU3Y2UyM2M3MDdkODQyZDA5MjU2NyIsCiAgICAgICAgICAicmVzb2x2ZWQiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9yZWNlaXB0cy9jaGFpbl9mb2xkc19kZXJpdmVkLm5weiIKICAgICAgICB9LAogICAgICAgICJjYWNoZS9jaGFpbnMubnB6IjogewogICAgICAgICAgImJ5dGVzIjogNjUwMTQyLAogICAgICAgICAgImRldmljZSI6IDY0NTEyLAogICAgICAgICAgImlub2RlIjogNzQyNDI4MzYwLAogICAgICAgICAgImlzbGluayI6IHRydWUsCiAgICAgICAgICAibXRpbWVfbnMiOiAxNzkwNjM5MDA0MDMyOTA1MjM5LAogICAgICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9kYXRhL2NhY2hlL2NoYWlucy5ucHoiLAogICAgICAgICAgInJlZ2lzdGVyZWRfc2hhMjU2IjogIjM5YWJhNDZmYWQ3YjcxNzgzMGU5MDY4N2UxYzRhNjk5MTc5NDg2NGNkMmNmNjIxYjBiYmYzNDFmMmQ5YTc3NjYiLAogICAgICAgICAgInJlc29sdmVkIjogIi9ob21lL2dyaW5jaGVua29lZy9XdW5kZXJGdW5kZXIvY2FjaGUvY2hhaW5zLm5weiIKICAgICAgICB9LAogICAgICAgICJjYWNoZS90cmFpbl90YXJnZXRzLm5weiI6IHsKICAgICAgICAgICJieXRlcyI6IDE5MDkzMzU3OTAsCiAgICAgICAgICAiZGV2aWNlIjogNjQ1MTIsCiAgICAgICAgICAiaW5vZGUiOiA3NDI0Mjg1OTksCiAgICAgICAgICAiaXNsaW5rIjogdHJ1ZSwKICAgICAgICAgICJtdGltZV9ucyI6IDE3OTA3MDI3Mjk1NjUyNTYwMzYsCiAgICAgICAgICAicGF0aCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLTIwMjYxMDE4L2RhdGEvY2FjaGUvdHJhaW5fdGFyZ2V0cy5ucHoiLAogICAgICAgICAgInJlZ2lzdGVyZWRfc2hhMjU2IjogImVhZDFhMDVlN2FhNWFjM2QxYzEwY2RlNjU5ODZiZTlmNDhmZDM0MmNhZGI5NTMzMDYxMWU1OGMzYThmODE3ZTkiLAogICAgICAgICAgInJlc29sdmVkIjogIi9ob21lL2dyaW5jaGVua29lZy9XdW5kZXJGdW5kZXIvY2FjaGUvdHJhaW5fdGFyZ2V0cy5ucHoiCiAgICAgICAgfSwKICAgICAgICAiY2FjaGUvdmFsaWRfdGFyZ2V0cy5ucHoiOiB7CiAgICAgICAgICAiYnl0ZXMiOiAzMzcxODYxOTgsCiAgICAgICAgICAiZGV2aWNlIjogNjQ1MTIsCiAgICAgICAgICAiaW5vZGUiOiA3NDI0MjgwNzgsCiAgICAgICAgICAiaXNsaW5rIjogdHJ1ZSwKICAgICAgICAgICJtdGltZV9ucyI6IDE3OTA2Mzg5NDQ0NTI1OTY5OTksCiAgICAgICAgICAicGF0aCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLTIwMjYxMDE4L2RhdGEvY2FjaGUvdmFsaWRfdGFyZ2V0cy5ucHoiLAogICAgICAgICAgInJlZ2lzdGVyZWRfc2hhMjU2IjogImRiYjdhMWE0OGRkMGFmOWZhNTI0YTgwYzI2OTQwMDhhNTZjYjkxZDVkYWU4MGU4ZDBkNDk4NGMwYzU2MjI5NTUiLAogICAgICAgICAgInJlc29sdmVkIjogIi9ob21lL2dyaW5jaGVua29lZy9XdW5kZXJGdW5kZXIvY2FjaGUvdmFsaWRfdGFyZ2V0cy5ucHoiCiAgICAgICAgfSwKICAgICAgICAiZGF0YXNldHMvdHJhaW4ucGFycXVldCI6IHsKICAgICAgICAgICJieXRlcyI6IDI5MjQwMjE5MDExLAogICAgICAgICAgImRldmljZSI6IDY0NTEyLAogICAgICAgICAgImlub2RlIjogNzQzODM2MjQ5LAogICAgICAgICAgImlzbGluayI6IGZhbHNlLAogICAgICAgICAgIm10aW1lX25zIjogMTU3NzgzMzIwMDAwMDAwMDAwMCwKICAgICAgICAgICJwYXRoIjogIi9ob21lL2dyaW5jaGVua29lZy9jb21wdXRlci12aXNpb24taG9tZXdvcmstMjAyNjEwMTgvZGF0YS9kYXRhc2V0cy90cmFpbi5wYXJxdWV0IiwKICAgICAgICAgICJyZWdpc3RlcmVkX3NoYTI1NiI6ICI0YTc3NTY0NDlhNGJjZjI3MmNmNzE1YmI0MDVhMjE1Mjk4NzA0MTYwNWY2YzFjY2U2NTAyNDNhYjJhYmY0ZTgxIiwKICAgICAgICAgICJyZXNvbHZlZCI6ICIvaG9tZS9ncmluY2hlbmtvZWcvV3VuZGVyRnVuZGVyL2RhdGFzZXRzL3RyYWluLnBhcnF1ZXQiCiAgICAgICAgfSwKICAgICAgICAiZGF0YXNldHMvdmFsaWQucGFycXVldCI6IHsKICAgICAgICAgICJieXRlcyI6IDU2MTU1MDkxMjgsCiAgICAgICAgICAiZGV2aWNlIjogNjQ1MTIsCiAgICAgICAgICAiaW5vZGUiOiA3NDM4MzYyNTcsCiAgICAgICAgICAiaXNsaW5rIjogZmFsc2UsCiAgICAgICAgICAibXRpbWVfbnMiOiAxNTc3ODMzMjAwMDAwMDAwMDAwLAogICAgICAgICAgInBhdGgiOiAiL2hvbWUvZ3JpbmNoZW5rb2VnL2NvbXB1dGVyLXZpc2lvbi1ob21ld29yay0yMDI2MTAxOC9kYXRhL2RhdGFzZXRzL3ZhbGlkLnBhcnF1ZXQiLAogICAgICAgICAgInJlZ2lzdGVyZWRfc2hhMjU2IjogIjI4NjFlZTMwOTE0MDY1NzAyNjI2YTNhYzAzM2Q4MGQwNTYzY2RhY2FlYTUwYzY2N2ZjODI4YzJlZWI0MTg3ODIiLAogICAgICAgICAgInJlc29sdmVkIjogIi9ob21lL2dyaW5jaGVua29lZy9XdW5kZXJGdW5kZXIvZGF0YXNldHMvdmFsaWQucGFycXVldCIKICAgICAgICB9CiAgICAgIH0KICAgIH0KICB9LAogICJleHBvc3VyZSI6IHsKICAgICJkZXNjcmlwdGlvbiI6ICJmaXQtZXhjbHVkZWQgRVhQT1NFRCBkZXZlbG9wbWVudCBncm91cHMsIG5vdCBhIHByaXN0aW5lL3B1YmxpYyBob2xkb3V0IiwKICAgICJzY29wZSI6ICJPbmx5IHRoZSByZWdpc3RlcmVkIEYwL3NlZWQtMTAwMSBmaXZlLWFybSBwYW5lbCBhdCBmaXhlZCBlcG9jaCA1OyBhbGwgNDY4IGZpdC1leGNsdWRlZCBleHBvc2VkIGRldmVsb3BtZW50IGdyb3Vwcy4iLAogICAgIm9yZGVyZWRfZ3JvdXBfY291bnQiOiA0NjgsCiAgICAib3JkZXJlZF9tZW1iZXJzaGlwX3NoYTI1NiI6ICIxOGYwYzk1MTRkMjVlOTRjYmU1ZWIyNDYwNzg3OTMyNGJiODNkMTE3ZDk2MjY3MGQ0ZGRmOTczMzY0NWY0ZjBjIiwKICAgICJwcmlzdGluZV9vcl9wdWJsaWNfaG9sZG91dCI6IGZhbHNlLAogICAgImxpbWl0YXRpb25zIjogWwogICAgICAiVGhlIDQ2OCBGMCBncm91cHMgYXJlIGZpdC1leGNsdWRlZCBidXQgcmVwZWF0ZWRseSBleHBvc2VkIGRldmVsb3BtZW50IGV2aWRlbmNlLCBub3QgYSBmcmVzaCBvciBwdWJsaWMgaG9sZG91dC4iLAogICAgICAiQ29tcG9uZW50IGJvb3RzdHJhcCBpbnRlcnZhbHMgZGVzY3JpYmUgb25seSBncm91cC9jb21wb25lbnQgcmVzYW1wbGluZyBjb25kaXRpb25hbCBvbiB0aGVzZSBmaXhlZCBtb2RlbHM7IHRoZXkgZG8gbm90IGVzdGltYXRlIGZpdC1zZWVkIG9yIHB1YmxpYy10cmFuc2ZlciB1bmNlcnRhaW50eS4iLAogICAgICAiTm8gcHJlZGljdGlvbnMgd2VyZSBnZW5lcmF0ZWQgYnkgdGhpcyBzY29yZXI7IGl0IGNvbnN1bWVzIG9ubHkgdGhlIHJldmlld2VkIGZpeGVkIGVwb2NoLTUgZXhwb3J0LiIKICAgIF0KICB9LAogICJzY2FsYXJfdGFibGVzIjogewogICAgIm1ldHJpY19pZCI6ICJnbG9iYWxfd3BfY2xpcDJfdjEiLAogICAgIm9mZmljaWFsX21hc2siOiAiaXNfc2NvcmVkIEFORCAodCA+PSA5OSkiLAogICAgIm9yZGVyZWRfZ3JvdXBfY291bnQiOiA0NjgsCiAgICAiZGlzdGluY3RfY29tcG9uZW50X2NvdW50IjogMTQxLAogICAgImFybXMiOiB7CiAgICAgICJjb250cm9sIjogewogICAgICAgICJ0MCI6IDAuNjY1ODQ5NzUxOTM4Mjk0NCwKICAgICAgICAidDEiOiAwLjY4MzgxMDg0NDgzMzM4NTMsCiAgICAgICAgIndwIjogMC42NzQ4MzAyOTgzODU4Mzk5LAogICAgICAgICJzZWxlY3RlZF9yb3dzIjogODA4ODYzLAogICAgICAgICJtZXRyaWNfaWQiOiAiZ2xvYmFsX3dwX2NsaXAyX3YxIgogICAgICB9LAogICAgICAiZW1hIjogewogICAgICAgICJ0MCI6IDAuNjY1NzI5MzUyOTg4MzI3OSwKICAgICAgICAidDEiOiAwLjY4MzgyOTA4ODk2OTkzMzgsCiAgICAgICAgIndwIjogMC42NzQ3NzkyMjA5NzkxMzA4LAogICAgICAgICJzZWxlY3RlZF9yb3dzIjogODA4ODYzLAogICAgICAgICJtZXRyaWNfaWQiOiAiZ2xvYmFsX3dwX2NsaXAyX3YxIgogICAgICB9LAogICAgICAic3BhcnNlIjogewogICAgICAgICJ0MCI6IDAuNjY1ODQ4ODE0MTY1NTIyMSwKICAgICAgICAidDEiOiAwLjY4NDAyMDcwMjEyODIzOTgsCiAgICAgICAgIndwIjogMC42NzQ5MzQ3NTgxNDY4ODEsCiAgICAgICAgInNlbGVjdGVkX3Jvd3MiOiA4MDg4NjMsCiAgICAgICAgIm1ldHJpY19pZCI6ICJnbG9iYWxfd3BfY2xpcDJfdjEiCiAgICAgIH0sCiAgICAgICJubGwiOiB7CiAgICAgICAgInQwIjogMC42NjY0NjgwOTMxNjU5MTc4LAogICAgICAgICJ0MSI6IDAuNjg0NjY5MzQ1NzQzMjYyMiwKICAgICAgICAid3AiOiAwLjY3NTU2ODcxOTQ1NDU5MDEsCiAgICAgICAgInNlbGVjdGVkX3Jvd3MiOiA4MDg4NjMsCiAgICAgICAgIm1ldHJpY19pZCI6ICJnbG9iYWxfd3BfY2xpcDJfdjEiCiAgICAgIH0sCiAgICAgICJhbGx0aHJlZSI6IHsKICAgICAgICAidDAiOiAwLjY2NzAxMzA2NDU5MzY4NTcsCiAgICAgICAgInQxIjogMC42ODQ4MDQzNDM0NTkxNzM4LAogICAgICAgICJ3cCI6IDAuNjc1OTA4NzA0MDI2NDI5OCwKICAgICAgICAic2VsZWN0ZWRfcm93cyI6IDgwODg2MywKICAgICAgICAibWV0cmljX2lkIjogImdsb2JhbF93cF9jbGlwMl92MSIKICAgICAgfQogICAgfSwKICAgICJwYWlyZWRfY29udHJhc3RzX3ZzX2NvbnRyb2wiOiB7CiAgICAgICJlbWEiOiB7CiAgICAgICAgImNhbmRpZGF0ZV9taW51c19jb250cm9sIjogewogICAgICAgICAgImNhbmRpZGF0ZV93cCI6IDAuNjc0Nzc5MjIwOTc5MTMwOCwKICAgICAgICAgICJyZWZlcmVuY2Vfd3AiOiAwLjY3NDgzMDI5ODM4NTg0MDUsCiAgICAgICAgICAibWVhbl9kaWZmIjogLTUuMTA3NzQwNjcwOTczMTkzZS0wNSwKICAgICAgICAgICJib290c3RyYXBfc2QiOiA5LjA1ODI3MjQwODY3NDY3N2UtMDUsCiAgICAgICAgICAiY2k5NSI6IFsKICAgICAgICAgICAgLTAuMDAwMjU4NjQxNzQ3NTEyMDU2MTUsCiAgICAgICAgICAgIDguMDE5NTI3NzMxODUzNDMyZS0wNQogICAgICAgICAgXSwKICAgICAgICAgICJwX2xlXzAiOiAwLjY0ODUsCiAgICAgICAgICAibl9zZXF1ZW5jZXMiOiA0NjgsCiAgICAgICAgICAibl9jb21wb25lbnRzIjogMTQxLAogICAgICAgICAgIm5fYm9vdCI6IDIwMDAsCiAgICAgICAgICAic2VlZCI6IDI3MTgyOCwKICAgICAgICAgICJtZXRyaWNfaWQiOiAiZ2xvYmFsX3dwX2NsaXAyX3YxIiwKICAgICAgICAgICJjZW50ZXJlZF9tb21lbnRzIjogdHJ1ZQogICAgICAgIH0KICAgICAgfSwKICAgICAgInNwYXJzZSI6IHsKICAgICAgICAiY2FuZGlkYXRlX21pbnVzX2NvbnRyb2wiOiB7CiAgICAgICAgICAiY2FuZGlkYXRlX3dwIjogMC42NzQ5MzQ3NTgxNDY4ODEzLAogICAgICAgICAgInJlZmVyZW5jZV93cCI6IDAuNjc0ODMwMjk4Mzg1ODQwNSwKICAgICAgICAgICJtZWFuX2RpZmYiOiAwLjAwMDEwNDQ1OTc2MTA0MDc4NzE0LAogICAgICAgICAgImJvb3RzdHJhcF9zZCI6IDAuMDAwMjI2NTI4MDU5MjM1Mzc0MjcsCiAgICAgICAgICAiY2k5NSI6IFsKICAgICAgICAgICAgLTAuMDAwMzkxMzE5NjQwNTUzMDgzOSwKICAgICAgICAgICAgMC4wMDA1MDE4Mjk2Njk3NTEwNTU0CiAgICAgICAgICBdLAogICAgICAgICAgInBfbGVfMCI6IDAuMjg3LAogICAgICAgICAgIm5fc2VxdWVuY2VzIjogNDY4LAogICAgICAgICAgIm5fY29tcG9uZW50cyI6IDE0MSwKICAgICAgICAgICJuX2Jvb3QiOiAyMDAwLAogICAgICAgICAgInNlZWQiOiAyNzE4MjgsCiAgICAgICAgICAibWV0cmljX2lkIjogImdsb2JhbF93cF9jbGlwMl92MSIsCiAgICAgICAgICAiY2VudGVyZWRfbW9tZW50cyI6IHRydWUKICAgICAgICB9CiAgICAgIH0sCiAgICAgICJubGwiOiB7CiAgICAgICAgImNhbmRpZGF0ZV9taW51c19jb250cm9sIjogewogICAgICAgICAgImNhbmRpZGF0ZV93cCI6IDAuNjc1NTY4NzE5NDU0NTkwNSwKICAgICAgICAgICJyZWZlcmVuY2Vfd3AiOiAwLjY3NDgzMDI5ODM4NTg0MDUsCiAgICAgICAgICAibWVhbl9kaWZmIjogMC4wMDA3Mzg0MjEwNjg3NDk5OTQ4LAogICAgICAgICAgImJvb3RzdHJhcF9zZCI6IDAuMDAwNTUyOTk0NzA5NDAzMDM2OCwKICAgICAgICAgICJjaTk1IjogWwogICAgICAgICAgICAtMC4wMDAyOTI3MjQ4OTUzMzk4OTA5LAogICAgICAgICAgICAwLjAwMTg2OTU0MDM3Njg0MzQyNTcKICAgICAgICAgIF0sCiAgICAgICAgICAicF9sZV8wIjogMC4wODQ1LAogICAgICAgICAgIm5fc2VxdWVuY2VzIjogNDY4LAogICAgICAgICAgIm5fY29tcG9uZW50cyI6IDE0MSwKICAgICAgICAgICJuX2Jvb3QiOiAyMDAwLAogICAgICAgICAgInNlZWQiOiAyNzE4MjgsCiAgICAgICAgICAibWV0cmljX2lkIjogImdsb2JhbF93cF9jbGlwMl92MSIsCiAgICAgICAgICAiY2VudGVyZWRfbW9tZW50cyI6IHRydWUKICAgICAgICB9CiAgICAgIH0sCiAgICAgICJhbGx0aHJlZSI6IHsKICAgICAgICAiY2FuZGlkYXRlX21pbnVzX2NvbnRyb2wiOiB7CiAgICAgICAgICAiY2FuZGlkYXRlX3dwIjogMC42NzU5MDg3MDQwMjY0MjksCiAgICAgICAgICAicmVmZXJlbmNlX3dwIjogMC42NzQ4MzAyOTgzODU4NDA1LAogICAgICAgICAgIm1lYW5fZGlmZiI6IDAuMDAxMDc4NDA1NjQwNTg4NDc1NywKICAgICAgICAgICJib290c3RyYXBfc2QiOiAwLjAwMDU0NDQzMzQ3Njc3NTMzNDYsCiAgICAgICAgICAiY2k5NSI6IFsKICAgICAgICAgICAgNi44NzQ5MDcxMjI4NjM4MTdlLTA1LAogICAgICAgICAgICAwLjAwMjI0NTMwMDI5ODY1NTM1MjYKICAgICAgICAgIF0sCiAgICAgICAgICAicF9sZV8wIjogMC4wMTUsCiAgICAgICAgICAibl9zZXF1ZW5jZXMiOiA0NjgsCiAgICAgICAgICAibl9jb21wb25lbnRzIjogMTQxLAogICAgICAgICAgIm5fYm9vdCI6IDIwMDAsCiAgICAgICAgICAic2VlZCI6IDI3MTgyOCwKICAgICAgICAgICJtZXRyaWNfaWQiOiAiZ2xvYmFsX3dwX2NsaXAyX3YxIiwKICAgICAgICAgICJjZW50ZXJlZF9tb21lbnRzIjogdHJ1ZQogICAgICAgIH0KICAgICAgfQogICAgfSwKICAgICJ1bmNlcnRhaW50eV9zY29wZSI6ICJwYWlyZWQgZmVhdHVyZS1vbmx5IHN1Y2Nlc3Nvci1jb21wb25lbnQgYm9vdHN0cmFwIGNvbmRpdGlvbmFsIG9uIHRoZXNlIGZpdmUgZml4ZWQgZXBvY2gtNSBtb2RlbHM7IG5vdCBmaXQtc2VlZCBvciBwdWJsaWMtdHJhbnNmZXIgdW5jZXJ0YWludHkiLAogICAgImJvb3RzdHJhcCI6IHsKICAgICAgInJlcGxpY2F0ZXMiOiAyMDAwLAogICAgICAic2VlZCI6IDI3MTgyOCwKICAgICAgImNlbnRlcmVkX21vbWVudHMiOiB0cnVlCiAgICB9CiAgfSwKICAic3RvcF9zdGF0dXMiOiAiUEVORElOR19FWEFDVF9TVE9QIiwKICAic2NvcmVfZ2Fpbl9vcl9wdWJsaWNfdHJhbnNmZXJfY2xhaW0iOiBmYWxzZQp9Cg=='
ACCEPTANCE_SCHEMA = "fg19.first-cell-score-root-acceptance.v1"
ACCEPTANCE_STATUS = "ACCEPTED_FIXED_FIRST_CELL_DEVELOPMENT_SCORE_REPORT"
RESULT_SCHEMA = "fg19.first-cell-five-arm-streaming-score.v1"
RESULT_STATUS = "FIXED_FIRST_CELL_DEVELOPMENT_EVIDENCE_SCORED"
CGROUP_ROOT = Path("/sys/fs/cgroup")
MAX_ACCEPTANCE_BYTES = 1_000_000
MAX_COMMAND_CHUNK_BYTES = 4_096
MAX_CGROUP_DIRECTORIES = 128
MAX_CGROUP_ENTRIES_PER_DIRECTORY = 256
MAX_CGROUP_PROCS_FILE_BYTES = 32_768
MAX_CGROUP_PROCS_AGGREGATE_BYTES = 262_144
UNIT_PROPERTIES = ("Id", "LoadState", "ActiveState", "SubState", "MainPID") + (
    "InvocationID",
    "ExecMainCode",
    "ExecMainStatus",
    "Result",
    "ControlGroup",
)


def _is_hex(value: object, size: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == size
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_receipt(
    raw: bytes, byte_count: int, acceptance_sha: str, invocation: str
) -> tuple[str, dict[str, object]]:
    if (
        type(byte_count) is not int
        or not 0 < byte_count <= MAX_ACCEPTANCE_BYTES
        or not _is_hex(acceptance_sha, 64)
        or not _is_hex(invocation, 32)
    ):
        raise ValueError("score stop acceptance or invocation is unbound")
    digest = hashlib.sha256(raw).hexdigest()
    if (len(raw), digest) != (byte_count, acceptance_sha):
        raise ValueError("ROOT score acceptance bytes/SHA-256 differ")
    acceptance = json.loads(raw)
    if not isinstance(acceptance, dict):
        raise ValueError("ROOT score acceptance must be a JSON object")
    native = acceptance.get("native_terminal")
    result = acceptance.get("result")
    if not isinstance(native, dict) or not isinstance(result, dict):
        raise ValueError("ROOT score acceptance lacks native/result objects")
    if any(
        acceptance.get(key) != value
        for key, value in {
            "schema": ACCEPTANCE_SCHEMA,
            "status": ACCEPTANCE_STATUS,
            "unit": SCORE_UNIT,
            "invocation_id": invocation,
            "stop_status": "PENDING_EXACT_STOP",
        }.items()
    ):
        raise ValueError("ROOT score acceptance identity/status differs")
    if (
        native.get("active_state") != "active"
        or native.get("sub_state") != "exited"
        or type(native.get("main_pid")) is not int
        or native["main_pid"] != 0
        or type(native.get("exec_main_code")) is not int
        or native["exec_main_code"] != 1
        or type(native.get("exec_main_status")) is not int
        or native["exec_main_status"] != 0
    ):
        raise ValueError("ROOT acceptance lacks strict successful native exit facts")
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != RESULT_STATUS
        or type(result.get("bytes")) is not int
        or result["bytes"] <= 0
        or not _is_hex(result.get("sha256"), 64)
    ):
        raise ValueError("ROOT acceptance lacks the fixed score-result pin")
    if not isinstance(result.get("path"), str) or not result["path"]:
        raise ValueError("ROOT acceptance result path is missing")
    return digest, result


def _run_bounded(
    command: list[str], timeout: float, stdout_cap: int, stderr_cap: int
) -> tuple[int, str, str]:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError("systemd command pipes were unavailable")
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": stdout_cap, "stderr": stderr_cap}
    streams = {process.stdout: "stdout", process.stderr: "stderr"}
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("systemd command deadline exceeded")
                for key, _ in selector.select(remaining):
                    name = streams[key.fileobj]
                    capacity = limits[name] + 1 - len(outputs[name])
                    chunk = os.read(key.fileobj.fileno(), min(MAX_COMMAND_CHUNK_BYTES, capacity))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    outputs[name].extend(chunk)
                    if len(outputs[name]) > limits[name]:
                        raise ValueError("systemd command output exceeded its bound")
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=1)
        for stream in streams:
            stream.close()
    return (
        returncode,
        bytes(outputs["stdout"]).decode("utf-8", errors="strict"),
        bytes(outputs["stderr"]).decode("utf-8", errors="replace"),
    )


def _show(unit: str) -> dict[str, str]:
    if unit not in (SCORE_UNIT, "wnn-campaign.slice"):
        raise ValueError("unit outside the exact score stop scope")
    command = ["systemctl", "--user", "show", "--all", unit]
    for key in UNIT_PROPERTIES:
        command.extend(("-p", key))
    returncode, stdout, _ = _run_bounded(command, 12, 65_536, 4_096)
    if returncode:
        raise RuntimeError("exact systemd show failed")
    return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)


def _metadata_read(path: Path, byte_cap: int) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("cgroup metadata is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        if (opened.st_dev, opened.st_ino) != identity:
            raise ValueError("cgroup metadata changed before read")
        data = bytearray()
        while len(data) <= byte_cap:
            chunk = os.read(descriptor, min(MAX_COMMAND_CHUNK_BYTES, byte_cap + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
    finally:
        os.close(descriptor)
    if len(data) > byte_cap:
        raise ValueError("cgroup process metadata exceeded its byte cap")
    return bytes(data)


def _checked_cgroup(group: str, cgroup_root: Path) -> Path:
    if not group.startswith("/") or ".." in Path(group).parts:
        raise ValueError("campaign cgroup path is invalid")
    if not group.endswith("/wnn-campaign.slice"):
        raise ValueError("systemd cgroup is not the campaign parent slice")
    root = cgroup_root / group.lstrip("/")
    if root == cgroup_root or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("campaign cgroup is absent, redirected, or resolves to host root")
    return root


def _scan_populated(root: Path) -> dict[str, str]:
    pending, visited, aggregate = [root], 0, 0
    result: dict[str, str] = {}
    while pending:
        current = pending.pop()
        visited += 1
        if visited > MAX_CGROUP_DIRECTORIES:
            raise ValueError("campaign cgroup directory cap exceeded")
        raw = _metadata_read(current / "cgroup.procs", MAX_CGROUP_PROCS_FILE_BYTES)
        aggregate += len(raw)
        if aggregate > MAX_CGROUP_PROCS_AGGREGATE_BYTES:
            raise ValueError("campaign cgroup aggregate process cap exceeded")
        if raw.strip():
            result[str(current)] = raw.decode("ascii", errors="strict").strip()
        with os.scandir(current) as entries:
            for count, entry in enumerate(entries, start=1):
                if count > MAX_CGROUP_ENTRIES_PER_DIRECTORY or entry.is_symlink():
                    raise ValueError("campaign cgroup entry cap or symlink violation")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(current / entry.name)
                    if visited + len(pending) > MAX_CGROUP_DIRECTORIES:
                        raise ValueError("campaign cgroup directory cap exceeded")
    return result


def _populated() -> dict[str, str]:
    parent = _show("wnn-campaign.slice")
    if parent.get("Id") != "wnn-campaign.slice":
        raise ValueError("campaign parent slice identity differs")
    group = parent.get("ControlGroup", "")
    return _scan_populated(_checked_cgroup(group, CGROUP_ROOT))


def _require_before(properties: dict[str, str], invocation: str) -> None:
    expected = {
        "Id": SCORE_UNIT,
        "InvocationID": invocation,
        "ActiveState": "active",
        "SubState": "exited",
        "MainPID": "0",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
        "Result": "success",
    }
    if any(properties.get(key) != value for key, value in expected.items()):
        raise ValueError("score unit is not the accepted successful exited invocation")


def _require_after(properties: dict[str, str]) -> None:
    expected = {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "MainPID": "0",
        "InvocationID": "",
    }
    if any(properties.get(key) != value for key, value in expected.items()):
        raise RuntimeError("exact score stop outcome requires inspection; never retry")


def main() -> None:
    """Stop the score unit once after its exact ROOT result acceptance is durable."""
    raw = base64.b64decode(ACCEPTANCE_BASE64, validate=True)
    acceptance_sha, result = _validate_receipt(
        raw, ACCEPTANCE_BYTES, ACCEPTANCE_SHA256, SCORE_INVOCATION
    )
    before = _show(SCORE_UNIT)
    _require_before(before, SCORE_INVOCATION)
    populated_before = _populated()
    if populated_before:
        raise ValueError("campaign has populated descendants before score stop")
    stop_code, stop_stdout, stop_stderr = _run_bounded(
        ["systemctl", "--user", "stop", SCORE_UNIT], 30, 4_096, 4_096
    )
    after = _show(SCORE_UNIT)
    populated_after = _populated()
    _require_after(after)
    if stop_code != 0 or populated_after:
        raise RuntimeError(
            "exact score stop failed or campaign cgroup remained populated; never retry"
        )
    print(
        json.dumps(
            {
                "schema": "fg19-first-cell-score-exact-owned-stop.v1",
                "captured_utc": datetime.now(UTC).isoformat(),
                "unit": SCORE_UNIT,
                "invocation_id": SCORE_INVOCATION,
                "acceptance_sha256": acceptance_sha,
                "result_sha256": result["sha256"],
                "before": before,
                "after": after,
                "populated_before": populated_before,
                "populated_after": populated_after,
                "stop_returncode": stop_code,
                "stop_stdout": stop_stdout,
                "stop_stderr": stop_stderr,
                "scope": "One ROOT-accepted fixed development score unit only; stop status makes no score-quality or public-transfer claim.",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
