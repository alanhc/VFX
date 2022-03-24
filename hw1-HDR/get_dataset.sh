FILEID="1syWwiITM6RhnNOiF-4Plb8Up0bpqhKWX"
echo $FILEID
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=\
    $(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    'https://docs.google.com/uc?export=download&id=$FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID"\
     -O dataset.zip && rm -rf /tmp/cookies.txt
unzip dataset.zip
echo "done."