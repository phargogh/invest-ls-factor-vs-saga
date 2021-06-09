import fs from 'fs';
import path from 'path';

import yauzl from 'yauzl';

import { getLogger } from '../logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

export default function extractZipInplace(zipFilePath) {
  return new Promise((resolve) => {
    const extractToDir = path.dirname(zipFilePath);
    logger.info(`extracting ${zipFilePath}`);
    const options = {
      lazyEntries: true,
      autoClose: true,
    };
    yauzl.open(zipFilePath, options, (error, zipfile) => {
      if (error) throw error;
      zipfile.on('entry', (entry) => {
        const writePath = path.join(extractToDir, entry.fileName);
        // if entry is a directory
        if (/\/$/.test(entry.fileName)) {
          fs.mkdir(writePath, (e) => {
            if (e) {
              if (e.code === 'EEXIST') { } else throw e;
            }
            zipfile.readEntry();
          });
        } else {
          zipfile.openReadStream(entry, (err, readStream) => {
            if (err) throw err;
            readStream.on('end', () => {
              zipfile.readEntry();
            });
            // Sometimes an entry will be in a dir, where the
            // dir itself was *not* an entry, therefore we still need
            // to create the dir here.
            fs.mkdir(path.dirname(writePath), (e) => {
              if (e) {
                if (e.code === 'EEXIST') { } else throw e;
              }
              const writable = fs.createWriteStream(writePath);
              readStream.pipe(writable);
            });
          });
        }
      });
      zipfile.on('close', () => {
        resolve(true);
      });
      zipfile.readEntry();
    });
  });
}
