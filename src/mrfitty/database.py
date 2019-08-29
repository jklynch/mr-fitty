"""
The MIT License (MIT)

Copyright (c) 2015-2019 Joshua Lynch, Sarah Nicholas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the 'Software'), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from contextlib import contextmanager
from hashlib import sha256

from sqlalchemy import create_engine
from sqlalchemy import Column, Float, Integer, String
from sqlalchemy import ForeignKey, Table
from sqlalchemy import UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

FitBase = declarative_base()


class DBFile(FitBase):
    __tablename__ = "file"

    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True)
    digest = Column(String, unique=True)


fit_reference_spectrum = Table(
    "fit_reference_spectrum",
    FitBase.metadata,
    Column("fit_id", ForeignKey("fit.id"), primary_key=True),
    Column(
        "reference_spectrum_id", ForeignKey("reference_spectrum.id"), primary_key=True
    ),
)


class DBReferenceSpectrum(FitBase):
    __tablename__ = "reference_spectrum"
    __table_args__ = (
        UniqueConstraint(
            "spectrum_file_id", "start_energy", "end_energy", name="ref_1"
        ),
    )

    id = Column(Integer, primary_key=True)
    start_energy = Column(Float)
    end_energy = Column(Float)

    spectrum_file_id = Column(Integer, ForeignKey("file.id"))
    spectrum_file = relationship("DBFile")

    fits = relationship(
        "DBFit",
        secondary="fit_reference_spectrum",
        back_populates="reference_spectra",
        lazy="dynamic",
    )


class DBUnknownSpectrum(FitBase):
    __tablename__ = "unknown_spectrum"
    __table_args__ = (
        UniqueConstraint(
            "spectrum_file_id", "start_energy", "end_energy", name="unk_1"
        ),
    )

    id = Column(Integer, primary_key=True)
    start_energy = Column(Float)
    end_energy = Column(Float)

    spectrum_file_id = Column(Integer, ForeignKey("file.id"))
    spectrum_file = relationship("DBFile")

    fits = relationship("DBFit", back_populates="unknown_spectrum", lazy="dynamic")


class DBFit(FitBase):
    __tablename__ = "fit"

    id = Column(Integer, primary_key=True)
    start_energy = Column(Float)
    end_energy = Column(Float)
    sse = Column(Float)

    unknown_spectrum_id = Column(Integer, ForeignKey("unknown_spectrum.id"))
    unknown_spectrum = relationship("DBUnknownSpectrum", back_populates="fits")

    reference_spectra = relationship(
        "DBReferenceSpectrum", secondary="fit_reference_spectrum", back_populates="fits"
    )


@contextmanager
def session_scope(Session_):
    """Provide a transactional scope around a series of operations."""
    session = Session_()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


class FitDatabase:
    def __init__(self, url, **kwargs):
        self.url = url
        self.engine = create_engine(self.url, **kwargs)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        FitBase.metadata.create_all(self.engine)

    def get_session_ctx_mgr(self):
        return session_scope(self.Session)

    @staticmethod
    def get_file_digest(path):
        h = sha256()
        with open(path, mode="rb") as b:
            h.update(b.read())
        digest = h.hexdigest()
        return digest

    def insert_file(self, session, path):
        session.add(DBFile(path=path, digest=self.get_file_digest(path=path)))

    def insert_reference_spectrum(self, session, reference_spectrum):
        dbfile = (
            session.query(DBFile)
            .filter(DBFile.path == reference_spectrum.file_path)
            .one_or_none()
        )
        if dbfile is None:
            self.insert_file(session, path=reference_spectrum.file_path)
            dbfile = (
                session.query(DBFile)
                .filter(DBFile.path == reference_spectrum.file_path)
                .one()
            )
        else:
            pass

        dbspectrum = DBReferenceSpectrum(
            spectrum_file=dbfile,
            start_energy=reference_spectrum.data_df.index[0],
            end_energy=reference_spectrum.data_df.index[-1],
        )

        session.add(dbspectrum)
        return dbspectrum

    @staticmethod
    def query_reference_spectra(session, path):
        return (
            session.query(DBReferenceSpectrum)
            .join(DBFile)
            .filter(DBFile.path == path)
            .one()
        )

    def insert_unknown_spectrum(self, session, unknown_spectrum):
        dbfile = (
            session.query(DBFile)
            .filter(DBFile.path == unknown_spectrum.file_path)
            .one_or_none()
        )
        if dbfile is None:
            self.insert_file(session, path=unknown_spectrum.file_path)
            dbfile = (
                session.query(DBFile)
                .filter(DBFile.path == unknown_spectrum.file_path)
                .one()
            )
        else:
            pass

        dbspectrum = DBUnknownSpectrum(
            spectrum_file=dbfile,
            start_energy=unknown_spectrum.data_df.index[0],
            end_energy=unknown_spectrum.data_df.index[-1],
        )

        session.add(dbspectrum)
        return dbspectrum

    @staticmethod
    def query_unknown_spectra(session, path):
        """
        Unknown spectrum database records are unique by path.

        The start and end energies for these records are not necessarily the same as for fits.

        Parameters
        ----------
        session    database session
        path       path of unknown spectrum file

        Returns
        -------
        One instance of DBUnknownSpectrum
        """
        return (
            session.query(DBUnknownSpectrum)
            .join(DBFile)
            .filter(DBFile.path == path)
            .one()
        )

    def insert_fit(self, session, fit):
        """
        The associated reference and unknown spectra must be already in the database.

        Parameters
        ----------
        session    database session
        fit        instance of SpectrumFit
        """
        dbfit = DBFit(
            start_energy=fit.get_start_energy(),
            end_energy=fit.get_end_energy(),
            sse=fit.nss,
        )

        db_unknown_spectrum = self.query_unknown_spectra(
            session=session, path=fit.unknown_spectrum.file_path
        )
        dbfit.unknown_spectrum = db_unknown_spectrum

        for r in fit.reference_spectra_seq:
            db_ref = self.query_reference_spectra(session=session, path=r.file_path)
            dbfit.reference_spectra.append(db_ref)

        session.add(dbfit)

    def query_fits(self, session, unknown_spectrum):
        db_unknown_spectrum = self.query_unknown_spectra(
            session=session, path=unknown_spectrum.file_path
        )

        return (
            session.query(DBFit)
            .filter(DBFit.unknown_spectrum_id == db_unknown_spectrum.id)
            .order_by(DBFit.sse)
            .limit(10)
            .all()
        )
