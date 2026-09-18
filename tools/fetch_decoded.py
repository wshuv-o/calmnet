"""Download the DECODED exoskeleton EEG dataset (figshare 21185362).

14 able-bodied participants wearing a lower-limb exoskeleton, 27 EEG channels
at 200 Hz, gait motor imagery against relax, recorded in two separate weeks.
That is the closest public match to cohort A: same application, same contrast,
and a multi-week structure the drift claim needs.

Streams to disk with progress so a 2.1 GB transfer can be watched and resumed
rather than restarted.
"""
import os, sys, json, urllib.request, time

# E: not D:. This 2.1 GB download filled D: at 0 bytes free on 2026-09-18,
# which is shared with other projects on this machine. Large data lives on E:
# from now on; the repo keeps only results.
OUT = os.environ.get("DECODED_ZIP", r"E:\calmnet_data\decoded\dataset.zip")


def main():
    r = urllib.request.Request("https://api.figshare.com/v2/articles/21185362",
                               headers={"User-Agent": "Mozilla/5.0"})
    meta = json.loads(urllib.request.urlopen(r, timeout=60).read())
    f = meta["files"][0]
    url, size = f["download_url"], f["size"]
    have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    if have >= size:
        print("already complete: %.0f MB" % (have / 1e6))
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if have:
        req.add_header("Range", "bytes=%d-" % have)
        print("resuming at %.0f MB of %.0f MB" % (have / 1e6, size / 1e6))
    t0, n = time.time(), have
    with urllib.request.urlopen(req, timeout=120) as rsp, \
            open(OUT, "ab" if have else "wb") as out:
        while True:
            chunk = rsp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            n += len(chunk)
            el = time.time() - t0
            if el > 0:
                print("  %5.1f%%  %6.0f/%.0f MB  %4.1f MB/s" %
                      (100 * n / size, n / 1e6, size / 1e6,
                       (n - have) / 1e6 / el), flush=True)
    print("done: %.0f MB" % (os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
