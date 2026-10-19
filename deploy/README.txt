deploy/chillrag.service — run the service under systemd
========================================================

Edit the unit first — exactly two things:

  1. `/path/to/ChillRAG`  (three occurrences: WorkingDirectory, PATH, ExecStart)
  2. `User=`              — the account that owns .venv/, materials/ and .cache/
                            (Group= is left unset; systemd uses the login group)

Install (needs root, once):

    sudo cp deploy/chillrag.service /etc/systemd/system/chillrag.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now chillrag

Verify:

    systemctl status chillrag --no-pager
    curl -s http://127.0.0.1:9022/healthz

Then kill whatever manual instance is still holding port 9022 first, or the
unit will crash-loop on "address already in use" until it is gone.

What you get
------------
- Boots with the machine (WantedBy=multi-user.target), survives power cuts.
- Comes back 10s after any crash (Restart=always). `systemctl stop` still means
  stop — Restart= never fights an explicit stop.
- stdout/stderr go to the journal:

    journalctl -u chillrag -f                 # follow
    journalctl -u chillrag --since today      # today

  If the journal loses old logs across reboots on this distro, make it
  persistent: `sudo mkdir -p /var/log/journal && sudo systemctl restart systemd-journald`.

No root?
--------
A user unit works the same way, but the service dies at logout unless lingering
is on (that one call needs admin, once):

    mkdir -p ~/.config/systemd/user
    cp deploy/chillrag.service ~/.config/systemd/user/   # same edits
    systemctl --user daemon-reload
    systemctl --user enable --now chillrag
    sudo loginctl enable-linger <user>

One honest caveat
-----------------
The ingest job queue lives in process memory and a reboot forgets it. A corpus
whose parse was interrupted by a reboot re-queues on its own only if it had no
store yet (`/api/upload/ensure` re-queues when the folder is not prepared). If
the interrupted parse was an *incremental* one — the folder already had a store
— the old corpus keeps serving and the new documents sit un-ingested. Symptom:
upload/status reports finish, but the new file's content does not come back in
recalls. Fix without code changes: delete that folder's derived cache and let
ensure rebuild it:

    rm -rf .cache/<folder>            # parsed artifacts only; materials/ is untouched
    # then trigger the folder once (attach / ensure / upload page)

So: avoid rebooting while a corpus is mid-ingest (`/api/upload/status` reporting
`state: processing` means one is).
