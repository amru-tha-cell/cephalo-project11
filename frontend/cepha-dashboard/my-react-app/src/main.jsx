// import React, { useEffect, useMemo, useRef, useState } from 'react';
// import { createRoot } from 'react-dom/client';
// import './styles.css';

// const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
// //const API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
// async function api(path, options = {}) { const r = await fetch(API + path, options); if (!r.ok) throw new Error(await r.text() || `HTTP ${r.status}`); const ct = r.headers.get('content-type') || ''; return ct.includes('json') ? r.json() : r.text(); }
// const unwrap = (x, k) => Array.isArray(x) ? x : (x?.[k] ?? x?.data ?? []);
// const caseIdOf = x => x?.case_id ?? x?.id;

// function App() {
//     const [cases, setCases] = useState([]), [current, setCurrent] = useState(null), [landmarks, setLandmarks] = useState([]), [measurements, setMeasurements] = useState(null), [analytics, setAnalytics] = useState({ mre: null, error: null, uncertainty: null, beforeAfter: null }), [history, setHistory] = useState([]), [report, setReport] = useState(null), [image, setImage] = useState(null), [selected, setSelected] = useState(null), [dragging, setDragging] = useState(null), [busy, setBusy] = useState(false), [toast, setToast] = useState(''), [tab, setTab] = useState('analysis'), [file, setFile] = useState(null);
//     const imgRef = useRef(null);
//     const notify = m => { setToast(m); setTimeout(() => setToast(''), 2400) };
//     const loadCases = async () => { try { setCases(unwrap(await api('/api/cases/'), 'cases')) } catch (e) { notify(e.message) } };
//     useEffect(() => { loadCases() }, []);
//     const loadCase = async c => { const id = caseIdOf(c); if (!id) return; setBusy(true); try { setCurrent({ ...c, case_id: id }); const vals = await Promise.allSettled([api(`/api/landmarks/${id}`), api(`/api/measurements/${id}`), api(`/api/analytics/mre-sdr/${id}`), api(`/api/analytics/landmark-error/${id}`), api(`/api/analytics/uncertainty/${id}`), api(`/api/analytics/before-after/${id}`), api(`/api/cases/${id}/history`), api(`/api/reports/${id}`)]); const get = (i) => vals[i].status === 'fulfilled' ? vals[i].value : null; setLandmarks(unwrap(get(0), 'landmarks')); setMeasurements(get(1)); setAnalytics({ mre: get(2), error: get(3), uncertainty: get(4), beforeAfter: get(5) }); setHistory(unwrap(get(6), 'history')); setReport(get(7)); setSelected(null); setTab('analysis'); const d = get(0); if (d?.image_url) setImage(d.image_url); else if (c.image_url) setImage(c.image_url); } catch (e) { notify(e.message) } finally { setBusy(false) } };
//     const upload = async () => { if (!file) return notify('Choose an X-ray first.'); setBusy(true); try { const fd = new FormData(); fd.append('file', file); const r = await api('/api/upload/', { method: 'POST', body: fd }); await loadCases(); const id = caseIdOf(r); if (id) await loadCase({ ...r, case_id: id }); notify('X-ray uploaded'); } catch (e) { notify(e.message) } finally { setBusy(false) } };
//     const infer = async () => { if (!current) return notify('Select a case first.'); setBusy(true); try { await api(`/api/inference/run/${current.case_id}`, { method: 'POST' }); await loadCase(current); notify('Inference completed'); } catch (e) { notify(e.message) } finally { setBusy(false) } };
//     const saveLandmark = async (i) => { if (!current) return; const l = landmarks[i]; try { await api(`/api/landmarks/${current.case_id}/${i}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ x: l.x, y: l.y }) }); notify(`Landmark ${i + 1} corrected`) } catch (e) { notify(e.message) } };
//     const recalc = async () => { if (!current) return; setBusy(true); try { await api(`/api/measurements/${current.case_id}/recalculate`, { method: 'POST' }); setMeasurements(await api(`/api/measurements/${current.case_id}`)); notify('Measurements recalculated') } catch (e) { notify(e.message) } finally { setBusy(false) } };
//     const remove = async () => { if (!current || !confirm('Delete this case?')) return; try { await api(`/api/cases/${current.case_id}`, { method: 'DELETE' }); setCurrent(null); await loadCases(); notify('Case deleted') } catch (e) { notify(e.message) } };
//     const normalized = useMemo(() => landmarks.map((l, i) => ({ ...l, index: i, name: l.name || l.label || `Landmark ${i + 1}`, x: Number(l.x ?? l.x_coordinate ?? 0), y: Number(l.y ?? l.y_coordinate ?? 0), confidence: Number(l.confidence ?? l.conf ?? 0) })), [landmarks]);
//     const move = (e) => { if (dragging === null) return; const r = imgRef.current.getBoundingClientRect(); const x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)); const y = Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)); setLandmarks(ls => ls.map((l, i) => i === dragging ? { ...l, x, y, x_coordinate: x, y_coordinate: y, corrected: true } : l)) };
//     return <div className="app" onPointerMove={move} onPointerUp={() => { if (dragging !== null) { saveLandmark(dragging); setDragging(null) } }}>
//         <header className="topbar"><div className="brand"><div className="logo">◉</div><div><b>CephaLandmark AI</b><small>Interactive cephalometric analysis</small></div></div><div className="topmeta"><span className="pill"><i /> API {API.replace(/^https?:\/\//, '')}</span><span className="pill">{current ? `Case ${current.case_id}` : 'No case selected'}</span></div><div className="actions"><label className="btn"><input type="file" accept="image/*" onChange={e => setFile(e.target.files?.[0] || null)} />Upload X-ray</label><button className="btn primary" onClick={upload} disabled={!file || busy}>Upload</button><button className="btn primary" onClick={infer} disabled={!current || busy}>Run inference</button></div></header>
//         <main className="layout">
//             <aside className="left"><div className="section-title">CASES <span>{cases.length}</span></div><div className="cases">{cases.map(c => <button className={'case ' + (current && caseIdOf(c) === current.case_id ? 'active' : '')} key={caseIdOf(c)} onClick={() => loadCase(c)}><b>Case {caseIdOf(c)}</b><small>{c.status || c.created_at || 'Available'}</small></button>)}{!cases.length && <p className="muted">No cases yet.</p>}</div><div className="section-title lmhead">LANDMARKS <span>{normalized.length}</span></div><div className="landmarks">{normalized.map(l => <button key={l.index} className={'lm ' + (selected === l.index ? 'sel ' : '') + (l.corrected ? 'corrected' : '')} onClick={() => setSelected(l.index)}><i /><span>{l.name}</span><small>{l.corrected ? 'manual' : l.confidence ? `${(l.confidence * 100).toFixed(0)}%` : '—'}</small></button>)}</div><div className="legend"><span><i className="cyan" />AI detected</span><span><i className="amber" />Manually corrected</span><small>Drag points on the image to correct them.</small></div></aside>
//             <section className="center"><div className="canvasbar"><span>{file?.name || current?.filename || 'X-ray not loaded'}</span><span>{busy ? 'Processing…' : current ? 'Analysis workspace' : 'Upload an X-ray to begin'}</span></div><div className="canvas"><div className="film" ref={imgRef}>{image ? <img src={image} alt="X-ray" /> : <div className="placeholder"><div>◉</div><b>Radiograph workspace</b><span>Upload an X-ray to display the image and landmarks.</span></div>}{normalized.map(l => <button key={l.index} className={'marker ' + (l.corrected ? 'corrected ' : '') + (selected === l.index ? 'selected' : '')} style={{ left: `${l.x * 100}%`, top: `${l.y * 100}%` }} onPointerDown={e => { e.stopPropagation(); setSelected(l.index); setDragging(l.index) }} title={l.name}><span>{l.index + 1}</span></button>)}</div></div><div className="canvasfoot"><span><b>{formatMetric(analytics.mre)}</b> MRE</span><span><b>{formatMetric(analytics.mre?.sdr ?? analytics.mre?.SDR)}</b> SDR</span><span><b>{normalized.length}</b> landmarks</span></div></section>
//             <aside className="right">
//                 <div className="tabs">{[['analysis', 'Analysis'], ['analytics', 'Analytics'], ['history', 'History'], ['report', 'Report']].map(([k, v]) => <button className={tab === k ? 'active' : ''} onClick={() => setTab(k)} key={k}>{v}</button>)}</div>
//                 {tab === 'analysis' && <><Card title="LANDMARK DETAIL">{selected === null ? <p className="muted">Select a landmark from the list or image.</p> : <LandmarkDetail l={normalized[selected]} onChange={(field, val) => setLandmarks(ls => ls.map((x, i) => i === selected ? { ...x, [field]: Number(val), corrected: true } : x))} onSave={() => saveLandmark(selected)} />}</Card><Card title="MEASUREMENTS"><Measure data={measurements} /><button className="wide primaryBtn" onClick={recalc} disabled={!current || busy}>Recalculate measurements</button></Card><div className="mini"><Stat label="CORRECTED" value={normalized.filter(x => x.corrected).length} /><Stat label="AVG CONF." value={avgConf(normalized)} /></div></>}
//                 {tab === 'analytics' && <><Card title="MRE / SDR"><Json data={analytics.mre} /></Card><Card title="LANDMARK ERROR"><Json data={analytics.error} /></Card><Card title="UNCERTAINTY"><Json data={analytics.uncertainty} /></Card><Card title="BEFORE / AFTER"><Json data={analytics.beforeAfter} /></Card></>}
//                 {tab === 'history' && <Card title="CASE HISTORY"><Timeline data={history} /></Card>}
//                 {tab === 'report' && <Card title="ANALYSIS REPORT"><Json data={report} /><button className="wide danger" onClick={remove}>Delete case</button></Card>}
//             </aside>
//         </main><div className={'toast ' + (toast ? 'show' : '')}><i />{toast}</div>
//     </div>
// }
// function Card({ title, children }) { return <div className="card"><div className="cardtitle">{title}</div>{children}</div> }
// function LandmarkDetail({ l, onChange, onSave }) { return <><b className="detailname">{l.name}</b><small className="mono">#{l.index + 1}</small><div className="fields"><label>X<input type="number" value={l.x} onChange={e => onChange('x', e.target.value)} /></label><label>Y<input type="number" value={l.y} onChange={e => onChange('y', e.target.value)} /></label></div><div className="row"><span>Confidence</span><b>{l.confidence ? (l.confidence * 100).toFixed(1) + '%' : 'Manual'}</b></div><div className="bar"><i style={{ width: `${Math.min(100, (l.confidence || 1) * 100)}%` }} /></div><button className="wide" onClick={onSave}>Save correction</button></> }
// function Measure({ data }) { if (!data) return <p className="muted">No measurements available.</p>; return <div className="measure">{Object.entries(data).slice(0, 8).map(([k, v]) => <div key={k}><span>{pretty(k)}</span><b>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</b></div>)}</div> }
// function Json({ data }) { if (!data) return <p className="muted">No data available.</p>; return <div className="json">{Object.entries(data).map(([k, v]) => <div key={k}><span>{pretty(k)}</span><b>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</b></div>)}</div> }
// function Timeline({ data }) { if (!data?.length) return <p className="muted">No history available.</p>; return <div className="timeline">{data.map((x, i) => <div key={i}><i /><div><b>{x.action || x.event || `Event ${i + 1}`}</b><small>{x.timestamp || x.created_at || ''}</small><Json data={x} /></div></div>)}</div> }
// function Stat({ label, value }) { return <div className="card stat"><b>{value}</b><small>{label}</small></div> }
// const pretty = k => k.replaceAll('_', ' ').replace(/\b\w/g, m => m.toUpperCase());
// const avgConf = a => a.length ? (a.reduce((s, x) => s + (x.corrected ? 1 : x.confidence || 0), 0) / a.length * 100).toFixed(1) + '%' : '—';
// const formatMetric = x => x == null ? '—' : typeof x === 'object' ? (x.value ?? x.mre ?? x.MRE ?? '—') : String(x);
// createRoot(document.getElementById('root')).render(<App />);
import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API =
    import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';


// =====================================================
// API HELPER
// =====================================================

async function api(path, options = {}) {
    const response = await fetch(API + path, options);

    if (!response.ok) {
        throw new Error(
            (await response.text()) ||
            `HTTP ${response.status}`
        );
    }

    const contentType =
        response.headers.get('content-type') || '';

    return contentType.includes('json')
        ? response.json()
        : response.text();
}


const caseIdOf = x =>
    x?.case_id ?? x?.id;


const unwrap = (x, key) =>
    Array.isArray(x)
        ? x
        : (x?.[key] ?? x?.data ?? []);


// =====================================================
// LANDMARK RESPONSE HANDLER
// =====================================================

function extractLandmarks(data) {
    if (!data) return [];

    const predicted =
        Array.isArray(data.predicted)
            ? data.predicted
            : [];

    const corrected =
        Array.isArray(data.corrected)
            ? data.corrected
            : [];

    const correctedMap = new Map(
        corrected.map(c => [
            c.landmark_index,
            c
        ])
    );

    return predicted.map((p, i) => {

        const correction =
            correctedMap.get(
                p.landmark_index
            );

        if (correction) {
            return {
                ...p,
                x: Number(correction.x),
                y: Number(correction.y),
                corrected: true
            };
        }

        return {
            ...p,
            x: Number(p.x),
            y: Number(p.y),
            corrected: false
        };
    });
}


// =====================================================
// PROCESSED IMAGE URL
// =====================================================

function processedImageUrl(path) {

    if (!path) return null;

    // Convert Windows path to URL-style path.
    const normalized =
        String(path).replaceAll('\\', '/');

    const filename =
        normalized.split('/').pop();

    if (!filename) return null;

    return (
        `${API}/static/processed/` +
        encodeURIComponent(filename)
    );
}


// =====================================================
// APP
// =====================================================

function App() {

    const [cases, setCases] = useState([]);

    const [current, setCurrent] =
        useState(null);

    const [landmarks, setLandmarks] =
        useState([]);

    const [measurements, setMeasurements] =
        useState(null);

    const [analytics, setAnalytics] =
        useState({
            mre: null,
            error: null,
            uncertainty: null,
            beforeAfter: null
        });

    const [history, setHistory] =
        useState([]);

    const [report, setReport] =
        useState(null);

    // This is now ONLY the backend-generated image.
    const [image, setImage] =
        useState(null);

    const [selected, setSelected] =
        useState(null);

    const [busy, setBusy] =
        useState(false);

    const [toast, setToast] =
        useState('');

    const [tab, setTab] =
        useState('analysis');

    const [file, setFile] =
        useState(null);


    // =================================================
    // NOTIFICATION
    // =================================================

    const notify = message => {

        setToast(message);

        setTimeout(
            () => setToast(''),
            2400
        );
    };


    // =================================================
    // LOAD CASES
    // =================================================

    const loadCases = async () => {

        try {

            const data =
                await api('/api/cases/');

            setCases(
                unwrap(data, 'cases')
            );

        } catch (error) {

            notify(error.message);

        }
    };


    useEffect(() => {
        loadCases();
    }, []);


    // =================================================
    // LOAD CASE
    // =================================================

    const loadCase = async caseData => {

        const id =
            caseIdOf(caseData);

        if (!id) return;

        setBusy(true);

        try {

            setCurrent({
                ...caseData,
                case_id: id
            });


            const results =
                await Promise.allSettled([

                    api(
                        `/api/landmarks/${id}`
                    ),

                    api(
                        `/api/measurements/${id}`
                    ),

                    api(
                        `/api/analytics/mre-sdr/${id}`
                    ),

                    api(
                        `/api/analytics/landmark-error/${id}`
                    ),

                    api(
                        `/api/analytics/uncertainty/${id}`
                    ),

                    api(
                        `/api/analytics/before-after/${id}`
                    ),

                    api(
                        `/api/cases/${id}/history`
                    ),

                    api(
                        `/api/reports/${id}`
                    )
                ]);


            const get = index =>
                results[index].status ===
                'fulfilled'
                    ? results[index].value
                    : null;


            // -----------------------------------------
            // LANDMARKS
            // -----------------------------------------

            const landmarkData =
                get(0);

            setLandmarks(
                extractLandmarks(
                    landmarkData
                )
            );


            // -----------------------------------------
            // MEASUREMENTS
            // -----------------------------------------

            setMeasurements(
                get(1)
            );


            // -----------------------------------------
            // ANALYTICS
            // -----------------------------------------

            setAnalytics({

                mre: get(2),

                error: get(3),

                uncertainty: get(4),

                beforeAfter: get(5)

            });


            // -----------------------------------------
            // HISTORY
            // -----------------------------------------

            setHistory(
                unwrap(
                    get(6),
                    'history'
                )
            );


            // -----------------------------------------
            // REPORT
            // -----------------------------------------

            setReport(
                get(7)
            );


            setSelected(null);

            setTab('analysis');


            // -----------------------------------------
            // EXISTING PROCESSED IMAGE
            // -----------------------------------------

            // If the case already contains a processed
            // image path, try to display it.
            if (caseData.heatmap_overlay_path) {

                setImage(
                    processedImageUrl(
                        caseData.heatmap_overlay_path
                    )
                );
            }

        } catch (error) {

            notify(error.message);

        } finally {

            setBusy(false);

        }
    };


    // =================================================
    // UPLOAD
    // =================================================

    const upload = async () => {

        if (!file) {

            notify(
                'Choose an X-ray first.'
            );

            return;
        }

        setBusy(true);

        try {

            const formData =
                new FormData();

            formData.append(
                'file',
                file
            );


            const response =
                await api(
                    '/api/upload/',
                    {
                        method: 'POST',
                        body: formData
                    }
                );


            await loadCases();


            const id =
                caseIdOf(response);


            if (id) {

                const newCase = {
                    ...response,
                    case_id: id
                };

                await loadCase(
                    newCase
                );
            }


            notify(
                'X-ray uploaded'
            );

        } catch (error) {

            notify(error.message);

        } finally {

            setBusy(false);

        }
    };


    // =================================================
    // RUN INFERENCE
    // =================================================

    const infer = async () => {

        if (!current) {

            notify(
                'Select a case first.'
            );

            return;
        }


        setBusy(true);

        try {

            const result =
                await api(
                    `/api/inference/run/${current.case_id}`,
                    {
                        method: 'POST'
                    }
                );


            // =========================================
            // IMPORTANT:
            // Backend returns:
            //
            // heatmap_overlay_path
            //
            // We directly display that generated image.
            // =========================================

            if (
                result?.heatmap_overlay_path
            ) {

                const processedUrl =
                    processedImageUrl(
                        result.heatmap_overlay_path
                    );

                setImage(
                    processedUrl
                );
            }


            // Reload landmarks,
            // measurements,
            // analytics, etc.
            await loadCase(
                current
            );


            // loadCase may not know the newly
            // generated image path, so restore it.
            if (
                result?.heatmap_overlay_path
            ) {

                setImage(
                    processedImageUrl(
                        result.heatmap_overlay_path
                    )
                );
            }


            notify(
                'Inference completed'
            );

        } catch (error) {

            notify(error.message);

        } finally {

            setBusy(false);

        }
    };


    // =================================================
    // SAVE LANDMARK CORRECTION
    // =================================================

    const saveLandmark = async index => {

        if (!current) return;

        const landmark =
            landmarks[index];

        if (!landmark) return;


        try {

            await api(
                `/api/landmarks/${current.case_id}/${landmark.landmark_index ?? index}`,
                {
                    method: 'PUT',

                    headers: {
                        'Content-Type':
                            'application/json'
                    },

                    body: JSON.stringify({

                        x: Number(
                            landmark.x
                        ),

                        y: Number(
                            landmark.y
                        )

                    })
                }
            );


            notify(
                `Landmark ${index + 1} corrected`
            );

        } catch (error) {

            notify(error.message);

        }
    };


    // =================================================
    // RECALCULATE MEASUREMENTS
    // =================================================

    const recalculate =
        async () => {

            if (!current) return;

            setBusy(true);

            try {

                await api(
                    `/api/measurements/${current.case_id}/recalculate`,
                    {
                        method: 'POST'
                    }
                );


                const updated =
                    await api(
                        `/api/measurements/${current.case_id}`
                    );


                setMeasurements(
                    updated
                );


                notify(
                    'Measurements recalculated'
                );

            } catch (error) {

                notify(error.message);

            } finally {

                setBusy(false);

            }
        };


    // =================================================
    // DELETE CASE
    // =================================================

    const remove = async () => {

        if (
            !current ||
            !confirm(
                'Delete this case?'
            )
        ) {
            return;
        }


        try {

            await api(
                `/api/cases/${current.case_id}`,
                {
                    method: 'DELETE'
                }
            );


            setCurrent(null);

            setLandmarks([]);

            setImage(null);

            setMeasurements(null);

            await loadCases();


            notify(
                'Case deleted'
            );

        } catch (error) {

            notify(error.message);

        }
    };


    // =================================================
    // NORMALIZED LANDMARKS
    // =================================================

    const normalized =
        useMemo(
            () => landmarks.map(
                (landmark, index) => ({

                    ...landmark,

                    index,

                    landmark_index:
                        landmark.landmark_index ??
                        index,

                    name:
                        landmark.name ||
                        landmark.label ||
                        `Landmark ${index + 1}`,

                    x: Number(
                        landmark.x ??
                        landmark.x_coordinate ??
                        0
                    ),

                    y: Number(
                        landmark.y ??
                        landmark.y_coordinate ??
                        0
                    ),

                    confidence:
                        Number(
                            landmark.confidence ??
                            landmark.conf ??
                            0
                        )

                })
            ),

            [landmarks]
        );


    // =================================================
    // UI
    // =================================================

    return (

        <div className="app">


            {/* =========================================
                HEADER
            ========================================== */}

            <header className="topbar">

                <div className="brand">

                    <div className="logo">
                        ◉
                    </div>

                    <div>

                        <b>
                            CephaLandmark AI
                        </b>

                        <small>
                            Interactive cephalometric analysis
                        </small>

                    </div>

                </div>


                <div className="topmeta">

                    <span className="pill">

                        <i />

                        API{' '}
                        {API.replace(
                            /^https?:\/\//,
                            ''
                        )}

                    </span>


                    <span className="pill">

                        {
                            current
                                ? `Case ${current.case_id}`
                                : 'No case selected'
                        }

                    </span>

                </div>


                <div className="actions">

                    <label className="btn">

                        <input
                            type="file"
                            accept="image/*"
                            onChange={e =>
                                setFile(
                                    e.target.files?.[0] ||
                                    null
                                )
                            }
                        />

                        Upload X-ray

                    </label>


                    <button
                        className="btn primary"
                        onClick={upload}
                        disabled={
                            !file ||
                            busy
                        }
                    >
                        Upload
                    </button>


                    <button
                        className="btn primary"
                        onClick={infer}
                        disabled={
                            !current ||
                            busy
                        }
                    >
                        Run inference
                    </button>

                </div>

            </header>


            {/* =========================================
                MAIN LAYOUT
            ========================================== */}

            <main className="layout">


                {/* =====================================
                    LEFT
                ====================================== */}

                <aside className="left">

                    <div className="section-title">

                        CASES

                        <span>
                            {cases.length}
                        </span>

                    </div>


                    <div className="cases">

                        {cases.map(c => (

                            <button
                                key={
                                    caseIdOf(c)
                                }

                                className={
                                    'case ' +
                                    (
                                        current &&
                                        caseIdOf(c) ===
                                        current.case_id
                                            ? 'active'
                                            : ''
                                    )
                                }

                                onClick={() =>
                                    loadCase(c)
                                }
                            >

                                <b>
                                    Case{' '}
                                    {caseIdOf(c)}
                                </b>

                                <small>
                                    {
                                        c.status ||
                                        c.created_at ||
                                        'Available'
                                    }
                                </small>

                            </button>

                        ))}


                        {!cases.length && (

                            <p className="muted">
                                No cases yet.
                            </p>

                        )}

                    </div>


                    <div className="section-title lmhead">

                        LANDMARKS

                        <span>
                            {normalized.length}
                        </span>

                    </div>


                    <div className="landmarks">

                        {normalized.map(l => (

                            <button
                                key={l.index}

                                className={
                                    'lm ' +
                                    (
                                        selected ===
                                        l.index
                                            ? 'sel '
                                            : ''
                                    ) +
                                    (
                                        l.corrected
                                            ? 'corrected'
                                            : ''
                                    )
                                }

                                onClick={() =>
                                    setSelected(
                                        l.index
                                    )
                                }
                            >

                                <i />

                                <span>
                                    {l.name}
                                </span>

                                <small>

                                    {
                                        l.corrected
                                            ? 'manual'
                                            : l.confidence
                                                ? `${(
                                                    l.confidence *
                                                    100
                                                ).toFixed(0)}%`
                                                : '—'
                                    }

                                </small>

                            </button>

                        ))}

                    </div>


                    <div className="legend">

                        <span>
                            <i className="cyan" />
                            AI detected
                        </span>

                        <span>
                            <i className="amber" />
                            Manually corrected
                        </span>

                        <small>
                            Select a landmark to edit
                            its coordinates.
                        </small>

                    </div>

                </aside>


                {/* =====================================
                    CENTER
                ====================================== */}

                <section className="center">

                    <div className="canvasbar">

                        <span>

                            {
                                file?.name ||
                                current?.filename ||
                                'X-ray not loaded'
                            }

                        </span>


                        <span>

                            {
                                busy
                                    ? 'Processing…'
                                    : current
                                        ? 'Analysis workspace'
                                        : 'Upload an X-ray to begin'
                            }

                        </span>

                    </div>


                    <div className="canvas">

                        <div className="film">

                            {image ? (

                                /*
                                 * IMPORTANT:
                                 *
                                 * There are NO React
                                 * landmark markers here.
                                 *
                                 * This image already comes
                                 * from backend/data/processing.
                                 */

                                <img
                                    src={image}
                                    alt="Processed cephalometric X-ray"
                                    draggable={false}
                                    onError={() =>
                                        notify(
                                            'Could not load processed image'
                                        )
                                    }
                                />

                            ) : (

                                <div className="placeholder">

                                    <div>
                                        ◉
                                    </div>

                                    <b>
                                        Radiograph workspace
                                    </b>

                                    <span>
                                        Upload an X-ray and
                                        run inference to
                                        display the processed
                                        image.
                                    </span>

                                </div>

                            )}

                        </div>

                    </div>


                    <div className="canvasfoot">

                        <span>

                            <b>
                                {
                                    formatMetric(
                                        analytics.mre
                                    )
                                }
                            </b>

                            {' '}MRE

                        </span>


                        <span>

                            <b>
                                {
                                    formatSDR(
                                        analytics.mre
                                    )
                                }
                            </b>

                            {' '}SDR

                        </span>


                        <span>

                            <b>
                                {normalized.length}
                            </b>

                            {' '}landmarks

                        </span>

                    </div>

                </section>


                {/* =====================================
                    RIGHT
                ====================================== */}

                <aside className="right">

                    <div className="tabs">

                        {[
                            [
                                'analysis',
                                'Analysis'
                            ],
                            [
                                'analytics',
                                'Analytics'
                            ],
                            [
                                'history',
                                'History'
                            ],
                            [
                                'report',
                                'Report'
                            ]

                        ].map(
                            ([key, label]) => (

                                <button
                                    key={key}
                                    className={
                                        tab === key
                                            ? 'active'
                                            : ''
                                    }
                                    onClick={() =>
                                        setTab(key)
                                    }
                                >
                                    {label}
                                </button>

                            )
                        )}

                    </div>


                    {/* =================================
                        ANALYSIS TAB
                    ================================= */}

                    {tab === 'analysis' && (

                        <>

                            <Card
                                title="LANDMARK DETAIL"
                            >

                                {
                                    selected === null
                                        ? (

                                            <p className="muted">
                                                Select a landmark
                                                from the list.
                                            </p>

                                        )
                                        : (

                                            <LandmarkDetail
                                                l={
                                                    normalized[
                                                        selected
                                                    ]
                                                }

                                                onChange={(
                                                    field,
                                                    value
                                                ) =>
                                                    setLandmarks(
                                                        list =>
                                                            list.map(
                                                                (item, i) =>
                                                                    i ===
                                                                    selected
                                                                        ? {
                                                                            ...item,
                                                                            [field]:
                                                                                Number(
                                                                                    value
                                                                                ),
                                                                            corrected:
                                                                                true
                                                                        }
                                                                        : item
                                                            )
                                                    )
                                                }

                                                onSave={() =>
                                                    saveLandmark(
                                                        selected
                                                    )
                                                }
                                            />

                                        )
                                }

                            </Card>


                            <Card
                                title="MEASUREMENTS"
                            >

                                <Measure
                                    data={
                                        measurements
                                    }
                                />


                                <button
                                    className="wide primaryBtn"
                                    onClick={
                                        recalculate
                                    }
                                    disabled={
                                        !current ||
                                        busy
                                    }
                                >
                                    Recalculate measurements
                                </button>

                            </Card>


                            <div className="mini">

                                <Stat
                                    label="CORRECTED"
                                    value={
                                        normalized.filter(
                                            x =>
                                                x.corrected
                                        ).length
                                    }
                                />


                                <Stat
                                    label="AVG CONF."
                                    value={
                                        avgConf(
                                            normalized
                                        )
                                    }
                                />

                            </div>

                        </>

                    )}


                    {/* =================================
                        ANALYTICS TAB
                    ================================= */}

                    {tab === 'analytics' && (

                        <>

                            <Card title="MRE / SDR">

                                <Json
                                    data={
                                        analytics.mre
                                    }
                                />

                            </Card>


                            <Card title="LANDMARK ERROR">

                                <Json
                                    data={
                                        analytics.error
                                    }
                                />

                            </Card>


                            <Card title="UNCERTAINTY">

                                <Json
                                    data={
                                        analytics.uncertainty
                                    }
                                />

                            </Card>


                            <Card title="BEFORE / AFTER">

                                <Json
                                    data={
                                        analytics.beforeAfter
                                    }
                                />

                            </Card>

                        </>

                    )}


                    {/* =================================
                        HISTORY TAB
                    ================================= */}

                    {tab === 'history' && (

                        <Card title="CASE HISTORY">

                            <Timeline
                                data={history}
                            />

                        </Card>

                    )}


                    {/* =================================
                        REPORT TAB
                    ================================= */}

                    {tab === 'report' && (

                        <Card title="ANALYSIS REPORT">

                            <Json
                                data={report}
                            />


                            <button
                                className="wide danger"
                                onClick={remove}
                            >
                                Delete case
                            </button>

                        </Card>

                    )}

                </aside>

            </main>


            {/* =========================================
                TOAST
            ========================================== */}

            <div
                className={
                    'toast ' +
                    (
                        toast
                            ? 'show'
                            : ''
                    )
                }
            >

                <i />

                {toast}

            </div>

        </div>
    );
}


// =====================================================
// CARD
// =====================================================

function Card({
    title,
    children
}) {

    return (

        <div className="card">

            <div className="cardtitle">
                {title}
            </div>

            {children}

        </div>

    );
}


// =====================================================
// LANDMARK DETAIL
// =====================================================

function LandmarkDetail({
    l,
    onChange,
    onSave
}) {

    if (!l) {

        return (
            <p className="muted">
                Landmark not found.
            </p>
        );
    }


    return (

        <>

            <b className="detailname">
                {l.name}
            </b>


            <small className="mono">
                #{l.index + 1}
            </small>


            <div className="fields">

                <label>

                    X

                    <input
                        type="number"
                        value={l.x}
                        onChange={e =>
                            onChange(
                                'x',
                                e.target.value
                            )
                        }
                    />

                </label>


                <label>

                    Y

                    <input
                        type="number"
                        value={l.y}
                        onChange={e =>
                            onChange(
                                'y',
                                e.target.value
                            )
                        }
                    />

                </label>

            </div>


            <div className="row">

                <span>
                    Confidence
                </span>

                <b>

                    {
                        l.confidence
                            ? (
                                l.confidence *
                                100
                            ).toFixed(1) + '%'
                            : 'Manual'
                    }

                </b>

            </div>


            <div className="bar">

                <i
                    style={{
                        width:
                            `${Math.min(
                                100,
                                (
                                    l.confidence ||
                                    1
                                ) * 100
                            )}%`
                    }}
                />

            </div>


            <button
                className="wide"
                onClick={onSave}
            >
                Save correction
            </button>

        </>

    );
}


// =====================================================
// MEASUREMENTS
// =====================================================

function Measure({ data }) {

    if (!data) {

        return (
            <p className="muted">
                No measurements available.
            </p>
        );
    }


    if (Array.isArray(data)) {

        if (!data.length) {

            return (
                <p className="muted">
                    No measurements available.
                </p>
            );
        }


        return (

            <div className="measure">

                {data
                    .slice(0, 8)
                    .map(
                        (measurement, index) => (

                            <div key={index}>

                                <span>
                                    {
                                        measurement.name ||
                                        'Measurement'
                                    }
                                </span>


                                <b>

                                    {
                                        measurement.value ??
                                        '—'
                                    }

                                    {
                                        measurement.unit
                                            ? ` ${measurement.unit}`
                                            : ''
                                    }

                                </b>

                            </div>

                        )
                    )}

            </div>

        );
    }


    return (

        <div className="measure">

            {Object.entries(data)
                .slice(0, 8)
                .map(
                    ([key, value]) => (

                        <div key={key}>

                            <span>
                                {pretty(key)}
                            </span>

                            <b>

                                {
                                    typeof value ===
                                    'object'
                                        ? JSON.stringify(
                                            value
                                        )
                                        : String(value)
                                }

                            </b>

                        </div>

                    )
                )}

        </div>

    );
}


// =====================================================
// JSON DISPLAY
// =====================================================

function Json({ data }) {

    if (!data) {

        return (
            <p className="muted">
                No data available.
            </p>
        );
    }


    if (Array.isArray(data)) {

        return (

            <div className="json">

                {data.map(
                    (item, index) => (

                        <div key={index}>

                            <span>
                                {
                                    item?.name ||
                                    item?.landmark ||
                                    `Item ${index + 1}`
                                }
                            </span>

                            <b>

                                {
                                    typeof item ===
                                    'object'
                                        ? JSON.stringify(
                                            item
                                        )
                                        : String(item)
                                }

                            </b>

                        </div>

                    )
                )}

            </div>

        );
    }


    return (

        <div className="json">

            {Object.entries(data).map(
                ([key, value]) => (

                    <div key={key}>

                        <span>
                            {pretty(key)}
                        </span>

                        <b>

                            {
                                typeof value ===
                                'object'
                                    ? JSON.stringify(
                                        value
                                    )
                                    : String(value)
                            }

                        </b>

                    </div>

                )
            )}

        </div>

    );
}


// =====================================================
// HISTORY
// =====================================================

function Timeline({ data }) {

    if (!data?.length) {

        return (
            <p className="muted">
                No history available.
            </p>
        );
    }


    return (

        <div className="timeline">

            {data.map(
                (item, index) => (

                    <div key={index}>

                        <i />

                        <div>

                            <b>

                                {
                                    item.action ||
                                    item.event_type ||
                                    item.event ||
                                    `Event ${index + 1}`
                                }

                            </b>


                            <small>

                                {
                                    item.timestamp ||
                                    item.created_at ||
                                    ''
                                }

                            </small>


                            <Json
                                data={item}
                            />

                        </div>

                    </div>

                )
            )}

        </div>

    );
}


// =====================================================
// STAT
// =====================================================

function Stat({
    label,
    value
}) {

    return (

        <div className="card stat">

            <b>
                {value}
            </b>

            <small>
                {label}
            </small>

        </div>

    );
}


// =====================================================
// HELPERS
// =====================================================

const pretty = key =>
    String(key)
        .replaceAll('_', ' ')
        .replace(
            /\b\w/g,
            m => m.toUpperCase()
        );


const avgConf = list => {

    if (!list.length) {
        return '—';
    }


    const value =
        list.reduce(
            (sum, item) =>
                sum +
                (
                    item.corrected
                        ? 1
                        : item.confidence ||
                          0
                ),
            0
        ) /
        list.length *
        100;


    return (
        value.toFixed(1) +
        '%'
    );
};


const formatMetric = value => {

    if (value == null) {
        return '—';
    }


    if (
        typeof value ===
        'object'
    ) {

        return (
            value.mre_mm ??
            value.mre ??
            value.MRE ??
            value.value ??
            '—'
        );
    }


    return String(value);
};


const formatSDR = value => {

    if (!value) {
        return '—';
    }


    const sdr =
        value.sdr ??
        value.SDR;


    if (!sdr) {
        return '—';
    }


    if (
        typeof sdr ===
        'object'
    ) {

        return (
            sdr['2'] ??
            sdr['2.0'] ??
            sdr['2.5'] ??
            '—'
        );
    }


    return String(sdr);
};


// =====================================================
// START REACT
// =====================================================

createRoot(
    document.getElementById('root')
).render(
    <App />
);