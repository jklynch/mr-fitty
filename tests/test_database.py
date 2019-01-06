"""
The MIT License (MIT)

Copyright (c) 2015-2018 Joshua Lynch, Sarah Nicholas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from mrfitty.database import DBFile
from tests import test_spectrum_fit

import pytest
import sqlalchemy.exc


def test_insert_files(fit_db, arsenic_references):
    with fit_db.get_session_ctx_mgr() as session:
        for arsenic_reference in arsenic_references:
            fit_db.insert_file(session, arsenic_reference.file_path)

    with fit_db.get_session_ctx_mgr() as session:
        db_files = session.query(DBFile).all()

        assert len(db_files) == len(arsenic_references)

    with pytest.raises(sqlalchemy.exc.IntegrityError) as ex:
        with fit_db.get_session_ctx_mgr() as session:
            fit_db.insert_file(session, arsenic_references[0].file_path)


def test_insert_reference_spectra(fit_db, arsenic_references):
    # insert DBFile explicitly
    with fit_db.get_session_ctx_mgr() as session:
        fit_db.insert_file(session, arsenic_references[0].file_path)
        fit_db.insert_reference_spectrum(session, arsenic_references[0])

    with fit_db.get_session_ctx_mgr() as session:
        sp = fit_db.query_reference_spectra(
            session=session,
            path=arsenic_references[0].file_path
        )

    # insert DBFile implicitly
    with fit_db.get_session_ctx_mgr() as session:
        fit_db.insert_reference_spectrum(session, arsenic_references[1])

    with fit_db.get_session_ctx_mgr() as session:
        sp = fit_db.query_reference_spectra(
            session=session,
            path=arsenic_references[1].file_path
        )


def test_insert_unknown_spectra(fit_db, arsenic_unknowns):
    # insert DBFile explicitly
    with fit_db.get_session_ctx_mgr() as session:
        fit_db.insert_file(session, arsenic_unknowns[0].file_path)
        fit_db.insert_unknown_spectrum(session, arsenic_unknowns[0])

    with fit_db.get_session_ctx_mgr() as session:
        sp = fit_db.query_unknown_spectra(
            session=session,
            path=arsenic_unknowns[0].file_path
        )

    # insert DBFile implicitly
    with fit_db.get_session_ctx_mgr() as session:
        fit_db.insert_unknown_spectrum(session, arsenic_unknowns[1])

    with fit_db.get_session_ctx_mgr() as session:
        sp = fit_db.query_unknown_spectra(
            session=session,
            path=arsenic_unknowns[1].file_path
        )


def test_insert_query_fit(fit_db, arsenic_references, arsenic_unknowns):
    """
    Insert fits with 1, 2, 3 components.

    Fits must be based on files.

    """
    fit_1 = test_spectrum_fit.generate_spectrum_fit(
        reference_count=1,
        reference_spectra=arsenic_references,
        unknown_spectrum=arsenic_unknowns[0]
    )
    fit_2 = test_spectrum_fit.generate_spectrum_fit(
        reference_count=2,
        reference_spectra=arsenic_references,
        unknown_spectrum=arsenic_unknowns[0]
    )
    fit_3 = test_spectrum_fit.generate_spectrum_fit(
        reference_count=3,
        reference_spectra=arsenic_references,
        unknown_spectrum=arsenic_unknowns[0]
    )

    with fit_db.get_session_ctx_mgr() as session:
        for arsenic_reference in arsenic_references:
            fit_db.insert_file(session, arsenic_reference.file_path)
        for arsenic_unknown in arsenic_unknowns:
            fit_db.insert_file(session, arsenic_unknown.file_path)

    with fit_db.get_session_ctx_mgr() as session:
        for arsenic_reference in arsenic_references:
            fit_db.insert_reference_spectrum(session, arsenic_reference)
        for arsenic_unknown in arsenic_unknowns:
            fit_db.insert_unknown_spectrum(session, arsenic_unknown)

    with fit_db.get_session_ctx_mgr() as session:
        fit_db.insert_fit(session, fit_1)
        fit_db.insert_fit(session, fit_2)
        fit_db.insert_fit(session, fit_3)

    with fit_db.get_session_ctx_mgr() as session:
        dbfits = fit_db.query_fits(
            session=session,
            unknown_spectrum=arsenic_unknowns[0]
        )
        assert len(dbfits) == 3

        sorted_dbfits = sorted(dbfits, key=lambda f: len(f.reference_spectra))
        assert len(sorted_dbfits[0].reference_spectra) == 1
        assert len(sorted_dbfits[1].reference_spectra) == 2
        assert len(sorted_dbfits[2].reference_spectra) == 3
