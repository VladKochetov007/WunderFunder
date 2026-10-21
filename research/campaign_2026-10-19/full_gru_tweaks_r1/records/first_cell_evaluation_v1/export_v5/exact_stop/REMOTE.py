from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

UNIT = "computer-vision-homework-fg19-first-cell-export-v3.service"
INVOCATION = "78fa25d0abc44d199e3696d172dbc4f9"
ACCEPTANCE_BYTES = 8833
ACCEPTANCE_SHA256 = "5a9a99121f80eb7fa07e4591318bea81078ca8b2db1032f150cc33afcc728d0d"
ACCEPTANCE_BASE64 = "ewogICJzY2hlbWEiOiAiZmcxOS5maXJzdC1jZWxsLWV4cG9ydC1hY2NlcHRhbmNlLnYxIiwKICAic3RhdHVzIjogIkFDQ0VQVEVEX0ZJWEVEX0ZJVkVfUFJFRElDVElPTl9FWFBPUlRTX05PVF9TQ09SRUQiLAogICJjcmVhdGVkX3V0YyI6ICIyMDI2LTEwLTIxVDExOjA3OjQwLjkyNDcyNiswMDowMCIsCiAgInVuaXQiOiAiY29tcHV0ZXItdmlzaW9uLWhvbWV3b3JrLWZnMTktZmlyc3QtY2VsbC1leHBvcnQtdjMuc2VydmljZSIsCiAgImludm9jYXRpb25faWQiOiAiNzhmYTI1ZDBhYmM0NGQxOTllMzY5NmQxNzJkYmM0ZjkiLAogICJuYXRpdmVfdGVybWluYWwiOiB7CiAgICAiYWN0aXZlX3N0YXRlIjogImFjdGl2ZSIsCiAgICAic3ViX3N0YXRlIjogImV4aXRlZCIsCiAgICAibWFpbl9waWQiOiAwLAogICAgImV4ZWNfbWFpbl9jb2RlIjogMSwKICAgICJleGVjX21haW5fc3RhdHVzIjogMAogIH0sCiAgInNvdXJjZV9yZXZpZXciOiB7CiAgICAicGF0aCI6ICIvVXNlcnMvbmIvRG9jdW1lbnRzL3d1bmRlci9XdW5kZXJGdW5kZXItbWFpbnRlbmFuY2UvcmVzZWFyY2gvY2FtcGFpZ25fMjAyNi0xMC0xOS9mdWxsX2dydV90d2Vha3NfcjEvcHJlcGFyYXRpb24vZmlyc3RfY2VsbF9ldmFsdWF0aW9uX2V4cG9ydF92NC9FWFBPUlRfU09VUkNFX1JFVklFV19WNC5qc29uIiwKICAgICJieXRlcyI6IDg3MTIsCiAgICAic2hhMjU2IjogIjY0YzkzZDgyNzlhMmRhZWIxNWYyNzUwZTYzNjZhMGZlOTRhMjY4YzdiOGU5YTA0ZGJkYjI4ZTRhMzZhNTZlNzMiCiAgfSwKICAiYmluZGluZyI6IHsKICAgICJwYXRoIjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9wcmVwYXJhdGlvbi9maXJzdF9jZWxsX2V2YWx1YXRpb25fZXhwb3J0X3Y0L0VYUE9SVF9CSU5ESU5HX1Y0Lmpzb24iLAogICAgImJ5dGVzIjogMTU3OTYsCiAgICAic2hhMjU2IjogIjRhOGFlMjIyMDQ5NzFhYzA3ZDZlNWNkZmQ5ZjY1YTYyZTlmODQ0MDgzMTAzNWRiNzI3MWU0ZTBkYTNkODNhYmQiCiAgfSwKICAibGF1bmNoX2FjY2VwdGFuY2UiOiB7CiAgICAicGF0aCI6ICIvVXNlcnMvbmIvRG9jdW1lbnRzL3d1bmRlci9XdW5kZXJGdW5kZXItbWFpbnRlbmFuY2UvcmVzZWFyY2gvY2FtcGFpZ25fMjAyNi0xMC0xOS9mdWxsX2dydV90d2Vha3NfcjEvcmVjb3Jkcy9maXJzdF9jZWxsX2V2YWx1YXRpb25fdjEvZXhwb3J0X3Y1L0xBVU5DSF9BQ0NFUFRBTkNFLmpzb24iLAogICAgImJ5dGVzIjogMjIzOCwKICAgICJzaGEyNTYiOiAiYzQzZmY0YWU0Yzg4ZWIyMGM1YzQwOGU4MzA3ZDhmZTNiZGEzODJkOWVjNzY2NDhhYTUyZWYwZjE3ODFjNTY1MCIKICB9LAogICJsYXVuY2hfcmF3X2NhcHR1cmUiOiB7CiAgICAicGF0aCI6ICIvVXNlcnMvbmIvRG9jdW1lbnRzL3d1bmRlci9XdW5kZXJGdW5kZXItbWFpbnRlbmFuY2UvcmVzZWFyY2gvY2FtcGFpZ25fMjAyNi0xMC0xOS9mdWxsX2dydV90d2Vha3NfcjEvcmVjb3Jkcy9maXJzdF9jZWxsX2V2YWx1YXRpb25fdjEvZXhwb3J0X3Y1L1NURE9VVC5yYXcuanNvbiIsCiAgICAiYnl0ZXMiOiAzODUyOCwKICAgICJzaGEyNTYiOiAiYTEzZGNmN2EwMjBmMDgyZDIzNjdmMDA5YWRjM2NhYTI4ZDUzOTZlYTRjNWU4NTY2MzM4YTk2YWIzNDcwYTg5OCIKICB9LAogICJsYXVuY2hfcmF3X3RyYW5zcG9ydCI6IHsKICAgICJwYXRoIjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9yZWNvcmRzL2ZpcnN0X2NlbGxfZXZhbHVhdGlvbl92MS9leHBvcnRfdjUvVFJBTlNQT1JULmpzb24iLAogICAgImJ5dGVzIjogNzMyLAogICAgInNoYTI1NiI6ICJjMmNjMzNiYTQwNjJmNDg0ZmQ5MTQ2Zjg2MzMyYzk4MGNiODE5YWZmZTYyY2I2NTdjZDMxNjE0NzUyMzhjMTA1IgogIH0sCiAgImxhdW5jaF9yZWNlaXB0cyI6IHsKICAgICJBRE1JU1NJT04uanNvbiI6IHsKICAgICAgImJ5dGVzIjogMjQ2NjAsCiAgICAgICJzaGEyNTYiOiAiMjZhYzljMDhjOGJhNjMyYTgxNjM0MDc1ODRiOWZjNmVmOWRmZTAzNDViYzUzNTlkZDZmNWRkZDc0NTEwYWJhMSIKICAgIH0sCiAgICAiTEFVTkNILmpzb24iOiB7CiAgICAgICJieXRlcyI6IDQxMzAsCiAgICAgICJzaGEyNTYiOiAiM2U5ODQ3NjEwMDVjNmM1NmVmYjA1NThmNDMyOTdiNWQzMGMwYWUwZmIwN2ZhODU5NTg5NTFjNGViZjNlMmIzNyIKICAgIH0KICB9LAogICJzdGF0dXNfY2FwdHVyZV9yb290IjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9yZWNvcmRzL2ZpcnN0X2NlbGxfZXZhbHVhdGlvbl92MS9leHBvcnRfdjUvc3RhdHVzXzIwMjYxMDIxVDExMDdaIiwKICAibmF0aXZlX3Jhd19jYXB0dXJlIjogewogICAgInBhdGgiOiAiL1VzZXJzL25iL0RvY3VtZW50cy93dW5kZXIvV3VuZGVyRnVuZGVyLW1haW50ZW5hbmNlL3Jlc2VhcmNoL2NhbXBhaWduXzIwMjYtMTAtMTkvZnVsbF9ncnVfdHdlYWtzX3IxL3JlY29yZHMvZmlyc3RfY2VsbF9ldmFsdWF0aW9uX3YxL2V4cG9ydF92NS9zdGF0dXNfMjAyNjEwMjFUMTEwN1ovU1RET1VULnJhdy5qc29uIiwKICAgICJieXRlcyI6IDE0MzE4OCwKICAgICJzaGEyNTYiOiAiMDRlMTJiNjI0ZDMyNWFlOTNjMTFjMjQxYzljZDEwMTFjZWIyZWY3YWNkMGFlZDdkMDkyNTJhNzZhODM0YWIxYSIKICB9LAogICJuYXRpdmVfcmF3X3RyYW5zcG9ydCI6IHsKICAgICJwYXRoIjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9yZWNvcmRzL2ZpcnN0X2NlbGxfZXZhbHVhdGlvbl92MS9leHBvcnRfdjUvc3RhdHVzXzIwMjYxMDIxVDExMDdaL1RSQU5TUE9SVC5qc29uIiwKICAgICJieXRlcyI6IDYwOSwKICAgICJzaGEyNTYiOiAiMmVjYTI5YTNmNmRiZTA3NjFlNWJjOTc0NDExMGIwMTFkOTc4YmZmZjVmODI3MTMwYzRjZjE2ZDlhZWM4OGQ2MyIKICB9LAogICJzdGF0dXNfc3RkZXJyX2NhcHR1cmUiOiB7CiAgICAicGF0aCI6ICIvVXNlcnMvbmIvRG9jdW1lbnRzL3d1bmRlci9XdW5kZXJGdW5kZXItbWFpbnRlbmFuY2UvcmVzZWFyY2gvY2FtcGFpZ25fMjAyNi0xMC0xOS9mdWxsX2dydV90d2Vha3NfcjEvcmVjb3Jkcy9maXJzdF9jZWxsX2V2YWx1YXRpb25fdjEvZXhwb3J0X3Y1L3N0YXR1c18yMDI2MTAyMVQxMTA3Wi9TVERFUlIucmF3LnR4dCIsCiAgICAiYnl0ZXMiOiAwLAogICAgInNoYTI1NiI6ICJlM2IwYzQ0Mjk4ZmMxYzE0OWFmYmY0Yzg5OTZmYjkyNDI3YWU0MWU0NjQ5YjkzNGNhNDk1OTkxYjc4NTJiODU1IgogIH0sCiAgInN0YXR1c19yZWFkZXJfYm9keSI6IHsKICAgICJwYXRoIjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9yZWNvcmRzL2ZpcnN0X2NlbGxfZXZhbHVhdGlvbl92MS9leHBvcnRfdjUvc3RhdHVzXzIwMjYxMDIxVDExMDdaL1JFTU9URS5weSIsCiAgICAiYnl0ZXMiOiAxNDg4MywKICAgICJzaGEyNTYiOiAiNGUzNWU2MjExZjY2OGFkNmFlNGY5NDY2MTE5Zjk5NTQ0ZjQyZTNmY2E0NWJhOTEzMzdjNjA5OGRhNTRmZjMyOSIKICB9LAogICJzdGF0dXNfcmVhZGVyX2FzdF9iaW5kaW5nIjogewogICAgInBhdGgiOiAiL1VzZXJzL25iL0RvY3VtZW50cy93dW5kZXIvV3VuZGVyRnVuZGVyLW1haW50ZW5hbmNlL3Jlc2VhcmNoL2NhbXBhaWduXzIwMjYtMTAtMTkvZnVsbF9ncnVfdHdlYWtzX3IxL3JlY29yZHMvZmlyc3RfY2VsbF9ldmFsdWF0aW9uX3YxL2V4cG9ydF92NS9zdGF0dXNfMjAyNjEwMjFUMTEwN1ovQVNUX0JJTkRJTkcuanNvbiIsCiAgICAiYnl0ZXMiOiA4NTgsCiAgICAic2hhMjU2IjogIjFjMjIwOGYxOTllZDdkOTg3NWJkZjg4ZmU1ZDgyM2Y5YTkxY2MyZjc2NGU1MTBmOGEwNzJjZGNmMzJlNTU2NjYiCiAgfSwKICAic3RhdHVzX2RyaXZlcl9yZXZpZXciOiB7CiAgICAicGF0aCI6ICIvVXNlcnMvbmIvRG9jdW1lbnRzL3d1bmRlci9XdW5kZXJGdW5kZXItbWFpbnRlbmFuY2UvcmVzZWFyY2gvY2FtcGFpZ25fMjAyNi0xMC0xOS9mdWxsX2dydV90d2Vha3NfcjEvcHJlcGFyYXRpb24vZmlyc3RfY2VsbF9ldmFsdWF0aW9uX2FkbWlzc2lvbl92MS9ST09UX0NBUFRVUkVfRVhQT1JUX1NUQVRVU19TT1VSQ0VfUkVWSUVXX1Y1Lmpzb24iLAogICAgImJ5dGVzIjogMzU2NCwKICAgICJzaGEyNTYiOiAiZDIxNGY3NzVhYTcxZGU5NjJiOWFmNWIxYjU3MDdjY2IzMzliNTEzMmFlNjdlMDA0ZTA4MmY1ZmMyMjA0MGVjZCIKICB9LAogICJzdGF0dXNfZHJpdmVyX3Jldmlld19waW5fY29ycmVjdGlvbiI6IHsKICAgICJwYXRoIjogIi9Vc2Vycy9uYi9Eb2N1bWVudHMvd3VuZGVyL1d1bmRlckZ1bmRlci1tYWludGVuYW5jZS9yZXNlYXJjaC9jYW1wYWlnbl8yMDI2LTEwLTE5L2Z1bGxfZ3J1X3R3ZWFrc19yMS9wcmVwYXJhdGlvbi9maXJzdF9jZWxsX2V2YWx1YXRpb25fYWRtaXNzaW9uX3YxL1JPT1RfQ0FQVFVSRV9FWFBPUlRfU1RBVFVTX1Y1X1JFVklFV19QSU5fQ09SUkVDVElPTl9WMS5qc29uIiwKICAgICJieXRlcyI6IDIxNTcsCiAgICAic2hhMjU2IjogIjlkMTczN2U5YTEzZmU5YTVkN2JhODFhNjc1MzE5OGMwNDczMDFjYmQ4MGYyNTliYjJhMDFiYzIyNTY0ZWMyYWIiCiAgfSwKICAibWFuaWZlc3RfY2FwdHVyZSI6IHsKICAgICJieXRlcyI6IDIzMTMwLAogICAgInNoYTI1NiI6ICI2YjkzY2FiZjBlMGI0NDUwYTExODIzNjgyZGZlYmE5Mzk2MDc2ZWZiYzdiNGEyMDMzNjczZmM0NmI3ZjA0YzIzIgogIH0sCiAgIm1lbWJlcnNoaXBfc2hhMjU2IjogIjE4ZjBjOTUxNGQyNWU5NGNiZTVlYjI0NjA3ODc5MzI0YmI4M2QxMTdkOTYyNjcwZDRkZGY5NzMzNjQ1ZjRmMGMiLAogICJldmFsdWF0aW9uX2dyb3VwcyI6IDQ2OCwKICAib3V0cHV0cyI6IHsKICAgICJjb250cm9sIjogewogICAgICAicHJlZGljdGlvbiI6IHsKICAgICAgICAiYnl0ZXMiOiA3NDg4MDEyOCwKICAgICAgICAiZGF0YV9ieXRlcyI6IDc0ODgwMDAwLAogICAgICAgICJkdHlwZSI6ICJmbG9hdDMyLWxpdHRsZS1lbmRpYW4iLAogICAgICAgICJucHlfaGVhZGVyX2J5dGVzIjogMTI4LAogICAgICAgICJwYXRoIjogInByZWRpY3Rpb25zLm5weSIsCiAgICAgICAgInNoYTI1NiI6ICI2YWYyYjAxODM3ZTc4MmY0YThkMzM0ZWE5MDc4OWU2OTQzMWI2ZDA1OTAzMGFhNWNkYWI5N2FmYjBmZmJmOGM2IiwKICAgICAgICAic2hhcGUiOiBbCiAgICAgICAgICA0NjgsCiAgICAgICAgICAyMDAwMCwKICAgICAgICAgIDIKICAgICAgICBdCiAgICAgIH0sCiAgICAgICJjaGVja3BvaW50IjogewogICAgICAgICJieXRlcyI6IDQyNzk5NzAsCiAgICAgICAgInJlbGF0aXZlX3BhdGgiOiAiYXJ0aWZhY3RzL3J1bnMvZmcxOV9jaGlsZF9jb250cm9sX2YwX3MxMDAxX2ZpdF92MS9tb2RlbF9lcDUucHQiLAogICAgICAgICJzaGEyNTYiOiAiZDNmMjVlNTJhYTU5MDAyOTU5M2U1YWY2ZWEwNmM2ZDRmNDYwYmM2MTMxZmZhODVkNGNmMWNiMDljMzZiM2IzZiIsCiAgICAgICAgImxvYWRlZF9zaGEyNTYiOiAiZDNmMjVlNTJhYTU5MDAyOTU5M2U1YWY2ZWEwNmM2ZDRmNDYwYmM2MTMxZmZhODVkNGNmMWNiMDljMzZiM2IzZiIKICAgICAgfQogICAgfSwKICAgICJlbWEiOiB7CiAgICAgICJwcmVkaWN0aW9uIjogewogICAgICAgICJieXRlcyI6IDc0ODgwMTI4LAogICAgICAgICJkYXRhX2J5dGVzIjogNzQ4ODAwMDAsCiAgICAgICAgImR0eXBlIjogImZsb2F0MzItbGl0dGxlLWVuZGlhbiIsCiAgICAgICAgIm5weV9oZWFkZXJfYnl0ZXMiOiAxMjgsCiAgICAgICAgInBhdGgiOiAicHJlZGljdGlvbnMubnB5IiwKICAgICAgICAic2hhMjU2IjogImVkY2QyMjAzODIyNzBlNzRhY2E5OTA0NWE5OGE0MzU3YzFhOGExMTJhOTliNjBhZDBhMjRhYzc4NmE1NmZjZTciLAogICAgICAgICJzaGFwZSI6IFsKICAgICAgICAgIDQ2OCwKICAgICAgICAgIDIwMDAwLAogICAgICAgICAgMgogICAgICAgIF0KICAgICAgfSwKICAgICAgImNoZWNrcG9pbnQiOiB7CiAgICAgICAgImJ5dGVzIjogNDI3OTk3MCwKICAgICAgICAicmVsYXRpdmVfcGF0aCI6ICJhcnRpZmFjdHMvcnVucy9mZzE5X2NoaWxkX2VtYV9mMF9zMTAwMV9maXRfdjEvbW9kZWxfZXA1LnB0IiwKICAgICAgICAic2hhMjU2IjogIjZlOTBiYzRkYmE4YWRkMGNhZjY3MDA5NzUxOWYwOGYzY2UyMzlkMGI2NGMzN2M0N2IzNzYwYWEyMmFmNDE4NjIiLAogICAgICAgICJsb2FkZWRfc2hhMjU2IjogIjZlOTBiYzRkYmE4YWRkMGNhZjY3MDA5NzUxOWYwOGYzY2UyMzlkMGI2NGMzN2M0N2IzNzYwYWEyMmFmNDE4NjIiCiAgICAgIH0KICAgIH0sCiAgICAic3BhcnNlIjogewogICAgICAicHJlZGljdGlvbiI6IHsKICAgICAgICAiYnl0ZXMiOiA3NDg4MDEyOCwKICAgICAgICAiZGF0YV9ieXRlcyI6IDc0ODgwMDAwLAogICAgICAgICJkdHlwZSI6ICJmbG9hdDMyLWxpdHRsZS1lbmRpYW4iLAogICAgICAgICJucHlfaGVhZGVyX2J5dGVzIjogMTI4LAogICAgICAgICJwYXRoIjogInByZWRpY3Rpb25zLm5weSIsCiAgICAgICAgInNoYTI1NiI6ICJjYjU2ZmNhZWEyNWU2OTUzZDhjZTg0NmM0ZWFkZWVmMWM4N2FiYTJhNjQzZTRkYTI1YzA0ODc3MGRlM2IyMWFiIiwKICAgICAgICAic2hhcGUiOiBbCiAgICAgICAgICA0NjgsCiAgICAgICAgICAyMDAwMCwKICAgICAgICAgIDIKICAgICAgICBdCiAgICAgIH0sCiAgICAgICJjaGVja3BvaW50IjogewogICAgICAgICJieXRlcyI6IDQyNzk5NzAsCiAgICAgICAgInJlbGF0aXZlX3BhdGgiOiAiYXJ0aWZhY3RzL3J1bnMvZmcxOV9jaGlsZF9kaWN0aW9uYXJ5X2YwX3MxMDAxX2ZpdF92MS9tb2RlbF9lcDUucHQiLAogICAgICAgICJzaGEyNTYiOiAiYjg0MGQ5YzgxZDQ2NDA1ZjJkMjQzOWI5NGFkNmVmMTVmZjg1MDFiYjE0NzE1ZWM3OTc5MmQ5ODBlZmJmNzE1MSIsCiAgICAgICAgImxvYWRlZF9zaGEyNTYiOiAiYjg0MGQ5YzgxZDQ2NDA1ZjJkMjQzOWI5NGFkNmVmMTVmZjg1MDFiYjE0NzE1ZWM3OTc5MmQ5ODBlZmJmNzE1MSIKICAgICAgfQogICAgfSwKICAgICJubGwiOiB7CiAgICAgICJwcmVkaWN0aW9uIjogewogICAgICAgICJieXRlcyI6IDc0ODgwMTI4LAogICAgICAgICJkYXRhX2J5dGVzIjogNzQ4ODAwMDAsCiAgICAgICAgImR0eXBlIjogImZsb2F0MzItbGl0dGxlLWVuZGlhbiIsCiAgICAgICAgIm5weV9oZWFkZXJfYnl0ZXMiOiAxMjgsCiAgICAgICAgInBhdGgiOiAicHJlZGljdGlvbnMubnB5IiwKICAgICAgICAic2hhMjU2IjogImNhYjZlNWIyZDIyNWM5MjM2ZjcyYjBiYjc0NzAzNGJlODJmYjVlM2Y0YjMxOWM4NjdkMjBmOTEzZTczNTcwZDgiLAogICAgICAgICJzaGFwZSI6IFsKICAgICAgICAgIDQ2OCwKICAgICAgICAgIDIwMDAwLAogICAgICAgICAgMgogICAgICAgIF0KICAgICAgfSwKICAgICAgImNoZWNrcG9pbnQiOiB7CiAgICAgICAgImJ5dGVzIjogNDI3OTk3MCwKICAgICAgICAicmVsYXRpdmVfcGF0aCI6ICJhcnRpZmFjdHMvcnVucy9mZzE5X2NoaWxkX25sbF9mMF9zMTAwMV9maXRfdjEvbW9kZWxfZXA1LnB0IiwKICAgICAgICAic2hhMjU2IjogImIxOWE2MTM2M2I0YjQ5Y2Q5OGQwMjc3ZjJmNzRmZDNhYTNkYjUwYmI2YzFmY2IzMTRiNWZhYjU3MWUyN2ZkNTgiLAogICAgICAgICJsb2FkZWRfc2hhMjU2IjogImIxOWE2MTM2M2I0YjQ5Y2Q5OGQwMjc3ZjJmNzRmZDNhYTNkYjUwYmI2YzFmY2IzMTRiNWZhYjU3MWUyN2ZkNTgiCiAgICAgIH0KICAgIH0sCiAgICAiYWxsdGhyZWUiOiB7CiAgICAgICJwcmVkaWN0aW9uIjogewogICAgICAgICJieXRlcyI6IDc0ODgwMTI4LAogICAgICAgICJkYXRhX2J5dGVzIjogNzQ4ODAwMDAsCiAgICAgICAgImR0eXBlIjogImZsb2F0MzItbGl0dGxlLWVuZGlhbiIsCiAgICAgICAgIm5weV9oZWFkZXJfYnl0ZXMiOiAxMjgsCiAgICAgICAgInBhdGgiOiAicHJlZGljdGlvbnMubnB5IiwKICAgICAgICAic2hhMjU2IjogImFmYWVjOGRhOGI3NWI1Y2JlYzc2MDVhYjlkOGFlZjFlMzVkYWMzMmUxMmQ0NDRhM2NiMjAzMmI1NmY2Njg1NzUiLAogICAgICAgICJzaGFwZSI6IFsKICAgICAgICAgIDQ2OCwKICAgICAgICAgIDIwMDAwLAogICAgICAgICAgMgogICAgICAgIF0KICAgICAgfSwKICAgICAgImNoZWNrcG9pbnQiOiB7CiAgICAgICAgImJ5dGVzIjogNDI3OTk3MCwKICAgICAgICAicmVsYXRpdmVfcGF0aCI6ICJhcnRpZmFjdHMvcnVucy9mZzE5X2NoaWxkX2FsbHRocmVlX2YwX3MxMDAxX2ZpdF92MS9tb2RlbF9lcDUucHQiLAogICAgICAgICJzaGEyNTYiOiAiNzA3MjA3NzA5OTc0NGU2OGZmMjNkZTNlMzkwYjE0NjkyN2MyM2RhMGUxYzU0ZjViYjE4ZWIyZmIwNDY3NmMwNyIsCiAgICAgICAgImxvYWRlZF9zaGEyNTYiOiAiNzA3MjA3NzA5OTc0NGU2OGZmMjNkZTNlMzkwYjE0NjkyN2MyM2RhMGUxYzU0ZjViYjE4ZWIyZmIwNDY3NmMwNyIKICAgICAgfQogICAgfQogIH0sCiAgImV4cG9zdXJlIjogImZpdC1leGNsdWRlZCBFWFBPU0VEIGRldmVsb3BtZW50IGdyb3Vwcywgbm90IGEgcHJpc3RpbmUvcHVibGljIGhvbGRvdXQiLAogICJxdWFsaXR5X3Jlc3VsdCI6IG51bGwsCiAgImFycmF5X3BheWxvYWRzX3JlYWRfb3JfaGFzaGVkX2J5X1JPT1QiOiBmYWxzZSwKICAic3RvcF9zdGF0dXMiOiAiUEVORElOR19FWEFDVF9TVE9QIgp9Cg=="


def show(unit: str) -> dict[str, str]:
    """Inspect only the new exact export unit or its campaign resource slice."""
    if unit not in (UNIT, "wnn-campaign.slice"):
        raise ValueError("unit outside exact export stop scope")
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            "--all",
            unit,
            "-p",
            "Id",
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "InvocationID",
            "-p",
            "ExecMainCode",
            "-p",
            "ExecMainStatus",
            "-p",
            "Result",
            "-p",
            "ControlGroup",
        ],
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("exact stop unit query failed")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def populated() -> dict[str, str]:
    """Bound the parent descendant inventory and reject any populated process."""
    parent = show("wnn-campaign.slice")
    group = parent.get("ControlGroup", "")
    if group in ("", "/") or not group.endswith("/wnn-campaign.slice"):
        raise ValueError("unscoped parent cgroup")
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    if root.resolve(strict=True) != root:
        raise ValueError("parent cgroup redirected")
    stack, visited, total = [root], 0, 0
    result = {}
    while stack:
        current = stack.pop()
        visited += 1
        if visited > 128:
            raise ValueError("cgroup directory cap exceeded")
        with os.scandir(current) as entries:
            for index, entry in enumerate(entries):
                if index >= 256:
                    raise ValueError("cgroup per-directory entry cap exceeded")
                if entry.is_symlink():
                    raise ValueError("cgroup directory contains symlink")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    if visited + len(stack) > 128:
                        raise ValueError("cgroup directory cap exceeded")
        fd = os.open(current / "cgroup.procs", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(32769)
        total += len(raw)
        if len(raw) > 32768 or total > 262144:
            raise ValueError("cgroup process bytes exceeded cap")
        if raw.strip():
            result[str(current)] = raw.decode("ascii")
    return result


def main() -> None:
    """Stop exactly one successful exited export only after ROOT acceptance is durable."""
    if len(INVOCATION) != 32 or any(c not in "0123456789abcdef" for c in INVOCATION):
        raise ValueError("actual export invocation unbound")
    raw = base64.b64decode(ACCEPTANCE_BASE64, validate=True)
    if (
        type(ACCEPTANCE_BYTES) is not int
        or ACCEPTANCE_BYTES <= 0
        or (len(raw), hashlib.sha256(raw).hexdigest()) != (ACCEPTANCE_BYTES, ACCEPTANCE_SHA256)
    ):
        raise ValueError("exact ROOT acceptance byte binding differs")
    acceptance = json.loads(raw)
    if (
        acceptance.get("schema"),
        acceptance.get("status"),
        acceptance.get("unit"),
        acceptance.get("invocation_id"),
        acceptance.get("quality_result"),
    ) != (
        "fg19.first-cell-export-acceptance.v1",
        "ACCEPTED_FIXED_FIVE_PREDICTION_EXPORTS_NOT_SCORED",
        UNIT,
        INVOCATION,
        None,
    ):
        raise ValueError("ROOT acceptance semantics differ")
    before = show(UNIT)
    if (
        before.get("Id"),
        before.get("InvocationID"),
        before.get("ActiveState"),
        before.get("SubState"),
        before.get("MainPID"),
        before.get("ExecMainCode"),
        before.get("ExecMainStatus"),
        before.get("Result"),
    ) != (UNIT, INVOCATION, "active", "exited", "0", "1", "0", "success"):
        raise ValueError("exact successful export is not exited before stop")
    populated_before = populated()
    if populated_before:
        raise ValueError("campaign has populated descendants before stop")
    stop = subprocess.run(
        ["systemctl", "--user", "stop", UNIT],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    after = show(UNIT)
    populated_after = populated()
    if (
        stop.returncode != 0
        or populated_after
        or (
            after.get("LoadState"),
            after.get("ActiveState"),
            after.get("SubState"),
            after.get("MainPID"),
            after.get("InvocationID"),
        )
        != ("not-found", "inactive", "dead", "0", "")
    ):
        raise RuntimeError("exact stop outcome requires inspection; never retry")
    print(
        json.dumps(
            {
                "schema": "fg19-first-cell-export-exact-owned-stop.v1",
                "captured_utc": datetime.now(UTC).isoformat(),
                "unit": UNIT,
                "invocation_id": INVOCATION,
                "acceptance_sha256": ACCEPTANCE_SHA256,
                "before": before,
                "after": after,
                "populated_before": populated_before,
                "populated_after": populated_after,
                "stop_returncode": stop.returncode,
                "stop_stdout": stop.stdout,
                "stop_stderr": stop.stderr,
                "scope": "Only the one successful exited export unit. All old native/support and foreign units excluded; no file or payload mutation.",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
